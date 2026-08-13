# Findings & Decisions

## 2026-08-12 nonlinear-neutraliser attenuation audit

- The supplied video is **"Frances Fulton | Wave Manipulation in Structures
  with Attached Nonlinear Neutralisers"**, published by the Isaac Newton
  Institute seminar-room channel. Search-engine indexing did not expose the
  video content reliably, so the source chain must be recovered from YouTube
  metadata/captions and then checked against primary papers or a thesis.
- YouTube metadata dates the 29:54 talk to 11 August 2026, identifies Fulton
  with the University of Auckland, and places it in INI programme `MWSW06`,
  *Multiple Wave Scattering in Locally Resonant Materials with Degrees of
  (Dis)Order*. The description points to INI seminar record `51186`; automatic
  English captions are available.
- The title makes this a plausible source for *passive nonlinear energy
  transfer/attenuation*, not yet an answer to subspace observability. The key
  audit question is whether the attachment changes reachable support, merely
  redirects energy among already reachable modes, or dissipates/temporarily
  captures response through amplitude-dependent coupling.
- No CHELATEDAI card, result, or novelty disposition changes from the video
  identity alone.
- The primary publication behind the talk is Fulton, Sorokin, and Abdi,
  *Elastic wave transmission through a semi-infinite rod with an attached
  damped nonlinear neutraliser* (2025). Its stated system is a host rod plus a
  local mass-spring-damper attachment with linear and cubic stiffness. The
  publication compares first- and second-order harmonic approximations (the
  latter retains first and third harmonics), stability, two incident waves,
  phase, amplitude, attachment-to-boundary distance, and reflection. These
  variables are materially narrower and better defined than the generic word
  "attenuation."
- Fulton's official research summary explicitly extends the single-attachment
  analysis toward multiple Duffing absorbers. The talk captions describe the
  key multi-attachment mechanism: a propagating wave changes amplitude and
  phase at each attachment, so nominally identical nonlinear neutralisers see
  different local forcing and therefore different amplitude-dependent
  effective resonance. That is a *sequential state-dependent impedance
  cascade*, not evidence for new spatial dimensions or discovery of an
  unobserved semantic direction.
- The leading CHELATEDAI transfer candidate is therefore a local auxiliary
  state coupled to graph propagation whose restoring/culling response depends
  nonlinearly on local residual amplitude. Its legitimacy will require a
  named nuisance projection, a useful-signal projection, an energy or gain
  functional, and a stability/passivity condition. Without those, phrases
  such as "subspace neutraliser" remain metaphor only.
- Early disposition: this mechanism may supply a concrete operator for
  `PRW-RCM1`-style reverberation control and a state-conditioned control for
  `PRW-DDF1`; it does not presently resolve `PRW-OBS1` off-span discovery,
  `PRW-SPU1` coordinate transport, or `PRW-COA1` coalition observability.
- The full 2026 talk is materially more cautious than its motivating picture.
  Fulton derives an iterative two-neutraliser model in which scattering at the
  second attachment is fed back to the first until a steady state is reached.
  One Duffing branch can yield three real solutions; composing two attachments
  can yield up to nine before duplicated branches proliferate under iteration.
  The presented two-neutraliser result shows possible broadening at a 20%
  reduction threshold, but also looping, isolas, jump states, unstable
  solutions, and substantially longer computation.
- The talk does **not** establish the proposed identical-component graded
  metamaterial as a general attenuation win. Fulton reports that, for the
  parameters shown, varying separation changed the response complexity but did
  not broaden the transmission dip. She explicitly leaves a broad damping and
  nonlinear-coefficient sweep, better duplicate-branch filtering, and higher
  harmonics as future work.
- The numerical comparison validates the analytical model mainly on stable
  branches and has some local deviation. The experiment is for a *single*
  neutraliser, not the two-neutraliser cascade: a clamped steel shim in an
  aluminium base, with magnet mass, attached to a steel rod. Increasing drive
  power shifts the observed dip upward as expected for a hardening Duffing
  response; the slide shows sizeable error bars and only qualitative/parameter
  consistency with theory.
- Consequently, the strongest transferable idea is not "stack more nonlinear
  layers." It is: attach small local state variables to a propagation path so
  their state-dependent impedance changes the gain seen by later sites, while
  explicitly measuring useful-signal retention, multistability, feedback
  branch count, and convergence cost. The talk supplies both the candidate
  mechanism and unusually relevant kill criteria.
- Terminology correction: Fulton's device has both linear and cubic stiffness,
  so it is closest to a Duffing-type nonlinear tuned vibration absorber or
  neutraliser. It should not be promoted to a classical essentially nonlinear
  energy sink with guaranteed one-way targeted energy transfer. Damping
  dissipates energy; undamped nonlinear stiffness primarily stores,
  redistributes, reflects, and frequency-converts it.
- The AI/ML ingredients are not individually novel. `GraphCON` (ICML 2022)
  already models node features as nonlinear controlled damped graph-coupled
  oscillators; `GRAND` and `GREAD` cover nonlinear diffusion and
  reaction-diffusion; `Wavy Transformer` (NeurIPS 2025) uses second-order wave
  dynamics in attention; `SLGNN` evolves amplitude and phase; ARMA graph
  filters and `GRAMA` supply recursive auxiliary/state-space filtering with
  dynamically selected coefficients.
- The RAG ingredients are also occupied. GNN-RAG learns query-relevant graph
  propagation; the final Findings of ACL 2026 CatRAG paper uses
  query-conditioned semantic edge weights and PPR-like traversal; and MemORAI
  combines provenance-enriched graph memory with query-conditioned weighted
  PageRank. A generic claim of "adaptive nonlinear attenuation," query-adaptive
  retrieval, or provenance-aware graph steering therefore collides with prior
  art.
- A deeper exact-combination pass found an even closer collision:
  Port-Hamiltonian Deep Graph Networks (ICLR 2025) already balance conservative
  and dissipative graph information flow with energy-based guarantees, while
  compositional port-Hamiltonian neural networks (L4DC 2023) already connect
  learned nonlinear spring-mass-damper subsystems and retain cyclo-passivity.
  Bracket-based graph dynamics (NeurIPS 2023) and compositional
  port-Hamiltonian distributed control (L4DC 2022) narrow this further. Thus
  neither passive/dissipative graph propagation, local dynamic controller
  memory, nor compositional nonlinear mechanical sidecars are novel
  ingredients here.
- The still-distinct, **novelty-unconfirmed** candidate is narrower: leave the
  primary evidence state untouched; attach resettable local sidecar states;
  couple them only through a provenance/certificate-derived nuisance map; use
  a passive hardening potential to make the response amplitude dependent; and
  require useful-evidence retention, lineage idempotence, multistability
  accounting, and query-time reversibility. No exact published collision was
  found in the bounded search, but component-wise prior art is dense and a
  specialist claim search remains mandatory before any novelty statement. The
  possible distinction is evidence-specific query reset, certificate-limited
  coupling, immutable provenance, and retrieval gates—not a new dynamics
  family.
- The formal candidate is now `PRW-RCM1-NLN`: a query-reset auxiliary state
  coupled to a fixed query-conditioned graph operator through an outcome-blind
  certificate map. With positive mass, damping, graph/attachment stiffness,
  and hardening cubic coefficients, its continuous-time storage function has
  nonpositive unforced derivative. This is a passivity screen, not a discrete
  solver guarantee or utility result.
- The finite graph model is a source-inspired Duffing attachment analogue,
  not a faithful transfer of Fulton's two-neutraliser procedure: it omits
  propagation delay, ordered reflection/scattering iteration, branch
  enumeration, and radiation-induced terms from the rod reduction.
- Independent mathematical review tightened that statement: the cancellation
  assumes a symmetric episode-fixed operator set, and nonincreasing
  semidefinite energy proves dissipativity, not bounded state or gain. A zero
  mode can drift unless the reachable state is coercively anchored or a
  detectability/quotient argument is supplied.
- Protected evidence is an explicit nullspace obligation on the attachment
  map, while the nuisance map may use declared provenance/dependency metadata
  but not REPORT labels or answer correctness. The sidecar can attenuate only
  directions it is told how to couple; it cannot expose an absent direction.
- The nullspace obligation alone is insufficient for exact protection: the
  host mass, damping, and propagation operators must also preserve the
  protected subspace. A sufficient fixed-episode condition is
  `C=C(I-P)` together with `[P,M]=[P,D]=[P,L]=0`. Stage A now states the
  stronger block-diagonal synthetic construction; live work must measure
  leakage.
- An independent scratch calculation found that the frozen high-amplitude
  forward Duffing maximum may lie at the 1.60 grid boundary. The official grid
  will not be widened. Endpoint maxima are now explicitly censored/unresolved,
  so a visually upward response cannot be promoted as a validated peak shift.
- The exact first-run Stage-A fixture is frozen in
  `docs/research/nonlinear-neutraliser-subspace-transfer-2026-08.md`. It
  checks the linear limit, unforced energy, Duffing hardening shift, protected
  leakage, full forward/reverse sweeps, and distributed attachments against
  equal-total-physical-coefficient controls. The co-located pair is an exact
  equivalence oracle, not a distinct efficacy control. Passing would establish
  only execution consistency on the frozen fixtures.
- Endpoint gain alone cannot test the source's most relevant multi-attachment
  claim. Before official execution, Stage A therefore adds report-only local
  mismatch amplitude, phase, third harmonic, and implied first-harmonic Duffing
  stiffness for every attachment. A distributed pair is "self-graded" only as
  a descriptive observation if its nominally identical attachments actually
  see different local responses; no utility claim follows from that difference.


## 2026-08-04 RB-14 bounded implementation and sanity evidence

- Implemented `observability_experiments.py` and the bounded runner
  `run_rb14_observability.py`. The module is NumPy/stdlib only, streaming for
  Gramian cells, deterministic by seed, atomic on artifact writes, and capped
  by explicit dimension/probe/work/byte/time budgets. The production retrieval
  path, model loading, corpus access, GPU, and concurrent fan-out were not used.
- Added 19 focused `unittest` cases; the focused RB-14 suite and the existing
  CRSV suite pass together: `50 tests`, `0.212s`, `OK`.
- The bounded runner wrote five artifacts under
  `artifacts/method-dev/rb14-observability/` with manifest digests. Every
  artifact has `status=COMPLETE`, `evidence_state=VALIDATED` for the exact
  synthetic sanity boundary, and `scientific_claim_status=UNCONFIRMED`.
- Re-ran the same bounded grid at seed `11` under
  `artifacts/method-dev/rb14-observability/seed-11/`; all five sanity maps were
  again true and every stored artifact digest recomputed exactly. This is
  seed robustness for the frozen fixture, not an independent scientific
  confirmation set.
- `PRW-OBS1`: in-span probes have zero planted off-span energy and never
  discover the hidden coordinate; the bounded full-rank control reaches rank
  16 and recovers it. This validates the identifiability guard, not a frontier
  policy advantage.
- `PRW-COA1`: singleton/top-k-below-degree masks remain exactly blind to pure
  pair/triad terms; eligible pair/triad masks recover the planted coefficient.
  This is a probe-support ceiling and does not upgrade the prior G2 result or
  establish utility/synergy.
- `PRW-CTX1`: naive singleton aggregation promotes the incompatible-witness
  fixture, while provenance/context/propensity-aware qualification returns
  `UNIDENTIFIED`; compatible shared-context positives promote; missing
  propensity support abstains. This is an exact quantifier guard, not a causal
  result from live data.
- `PRW-SPU1-TRANSPORT`: anisotropic coordinate transport produces measurable
  raw cosine drift, and a known inverse transport restores the latent metric
  to numerical tolerance. It is a confound control only; it does not show a
  new alignment method.
- `PRW-EK7-COALITION`: matched-token individual top-k misses the conjunctive
  query while a greedy complementary set recovers all required tokens under
  the same document budget. This is a deterministic surrogate only and does
  not pass the unchanged EK7 entry gate or prove RAG utility.
- Full repository discovery was intentionally stopped after a live process
  check showed roughly `709 MB` RSS, above the bounded single-slice envelope.
  The process emitted partial output (including a BCC-1 missing-manifest
  message) but no final summary; full-suite status is therefore **unverified**,
  not green. No OOM or repository mutation occurred.

## 2026-08-04 RB-14 implementation start

- The live repository has no exact RB-14 implementation or accepted result
  artifact. Existing CRSV code supplies reusable diagnostics, while the
  `prime_ring_rb10_contract.py` envelope is specific to prime-ring stages and
  should not be reused as if it were an observability contract.
- The first implementation slice will be a standalone NumPy/stdlib module with
  explicit budgets, deterministic seeds, streaming design-Gramian accounting,
  and atomic JSON artifacts. It will cover the OBS1 and COA1 sanity gates before
  any context, transport, or RAG integration work.
- User authorization now permits implementation and testing, but the prior
  G2 disposition remains fixed: pair/three-way residuals exist and static
  advantage is false. New code must test observability/support, not relabel
  interaction existence as a new mechanism result.
- A first planning patch attempt failed because its progress-file context was
  stale; no file was changed by that failure. The corrected plan update was
  applied after rereading the live surfaces.

## 2026-08-02 Open-span and coalitional observability queue reconciliation

- The new dimensionality intuition resolves into two distinct observability
  failures: untested directions outside the effective intervention span and
  untested interactions inside a lifted pair/triad feature space. Boundedness,
  deterministic oscillation, and embedding width are not the governing
  properties; intervention-design rank and conditioning are.
- Scalar decay, momentum, and perturbations restricted to an invariant known
  span cannot add observable rank. Conversely, bounded full-rank or
  persistently exciting designs can identify all declared linear directions.
- Singleton probes cannot identify pure pair effects because every pair
  feature is identically zero under top-1 activation. More generally, top-k
  probes cannot identify an otherwise unstructured pure interaction of degree
  greater than k.
- Existing RB-13 cards provide nearby but nonidentical work: `PRW-SPU1`
  supplies heterogeneous-space and Procrustes controls; `PRW-ISI1` tests
  nuisance invariance versus material interventions; `PRW-REV1` includes
  coordinate-map changes; and `PRW-EK7` is a survivor-only RAG integration.
  The new cards must reuse these controls and dependencies rather than claim
  the subjects are wholly unexplored.
- The attached extremal-graph result is an analogy, not AI evidence. Its useful
  warning is that singleton or aggregate compatibility need not establish a
  jointly compatible layered structure.
- This session is documentation-only. No protocol is frozen, no experiment
  code is implemented, and no test or scientific experiment is run.
- Read-only code/artifact reconciliation classifies all five exact RB-14 tests
  as `PROPOSED`, with these preservation boundaries:
  - no current sampler expands proposal support outside the declared span;
  - the completed `PRW-G2` algebraic artifact already validates necessary pair
    and strict three-way residuals while recording
    `static_control_advantage_established=false`; RB-14 must test the missing
    top-k observability ceiling rather than reimplement interaction arithmetic;
  - no current experiment tests the quantifier failure in which components
    are favorable under different contexts but lack one compatible joint
    witness;
  - query-swap, learned realignment, Procrustes, and representation bridges
    exist, but legacy quantitative drift evidence is
    `LEGACY_METRIC_LINEAGE_BLOCKED` and no accepted matched coordinate-transport
    confound test exists; and
  - recursive query decomposition fuses scores for individual documents but
    does not score document sets for complementary joint sufficiency, and no
    coalition-RAG result artifact exists.
- The existing RB-13 queue header says `DRAFT_QUEUE`, while the prior task-plan
  summary said `DRAFT_QUEUE_APPROVED`. Because RB-13 Section 1.1 is explicitly
  authoritative and the user requested no implementation or testing, the
  weaker `DRAFT_QUEUE` status governs this reconciliation.

## 2026-07-27 RB-13 evidence-kernel and masked-subplane queue design

- The planning-with-files catch-up helper returned no unsynchronized-session
  report.
- Before RB-13 documentation edits, Git reported
  `codex/prime-ring-onion-method-dev...origin/main [ahead 15]` with no changed
  worktree entries.
- This lane must not revive earlier geometric language as a mechanism by
  assertion. Each term will be retained only if it can be mapped to a typed
  operator and a falsifiable contrast.
- The literature audit places the broad evidence-first, bitemporal,
  contradiction-preserving memory architecture close to existing public work,
  especially Donto. Any CHELATEDAI contribution must therefore be framed as a
  measured delta, not as invention of the broad architecture.
- The current `ModelScopeMemoryStore`/`PersistentMemory` code is a bounded,
  mutable cache contract rather than an append-only evidence kernel: capacity
  can evict old entries, annotations mutate payloads, explicit promotion can
  bypass a promotion-status evaluation, absent episode status can be treated as
  promotable, and persistent keys can be overwritten or deleted. These are
  compatibility gaps for RB-13, not automatically bugs in the existing
  contract. `PRW-EK0` will characterize them before any redesign.
- The mathematical terms separate into different operator classes:
  - **deference** selects an answer/retrieve/verify/tool/human/abstain route;
  - **deflection** attenuates or redirects graph messages;
  - **reverberation** is bounded repeated propagation and cannot manufacture
    independent witnesses from cyclic walks;
  - **culling** is an ephemeral selection view, not evidence deletion; and
  - heterogeneous **subplanes** are local spaces linked by explicit
    compatibility maps, not extra physical dimensions.
- Support and refutation must remain separate channels. Scalar subtraction
  makes a highly contested claim observationally identical to an unknown claim.
  `PRW-BIL1` now tests an explicit four-state evidence bilattice, its two
  orders, meet/join operations, merge-order invariance, temporal correction,
  and finite fixed-point behavior against scalar and paired-boolean controls.
- Gelfand/Fomin and classical Euler-Lagrange are not incorrect. They apply to a
  declared smooth continuum functional. A finite graph/sheaf Laplacian energy
  instead has an ordinary finite-dimensional gradient/stationarity equation.
  Neither is a whole-system optimizer for binary masks, top-k selection, graph
  rewiring, variable-dimensional local spaces, provenance, or lattice-valued
  evidence state. `PRW-VAR1` therefore compares the smooth relaxation with
  exact, graph-cut/submodular, and proximal alternatives on small instances.
- The apparent subplane route survives the prior JO1 flattening result only if
  it contains a declared non-flat difference such as heterogeneous local
  dimensions, missing views, data-dependent routing, nonlinear/time-varying
  state, versioned coordinate maps, or a proved representation/decoder cost
  frontier. `PRW-SPU0` is the expected-null flat-reduction guard; `PRW-SPU1`
  is the information-matched heterogeneous-space test. Output equality is
  separate from resource equality: a factorized stack and a dense flat map can
  compute the same function with different bytes and operations, so both dense
  and best factorized-flat controls are required.
- Existing CRSV-1/LIR-1/SRS-1 code already covers principal angles, signed
  additive interference, commutators, finite-horizon amplification, ordered
  products, useful-signal atrophy, and a scale sweep on constructed examples.
  RB-13 freezes these as predecessor controls and does not rename them as new
  deference, deflection, culling, resonance, or variational evidence.
- The narrow candidate worth testing is the combination of a claim-typed,
  bitemporal, provenance-carrying, lattice-valued evidence kernel; paired
  nuisance-invariance/material-intervention checks; source-idempotent,
  correction-reversible query masks; reversible/versioned representation
  views; and action authority kept separate from factual support. The
  combination is still a conjecture and may reduce to ordinary temporal
  memory, copy-aware truth discovery, graph filtering, multi-view learning,
  truth maintenance, and selective prediction.
- Sixteen independently disposed draft cards are now specified in
  `docs/research/evidence-kernel-masked-subplane-experiment-queue-2026-07.md`.
  Their per-card protocols are not yet frozen, no card has run, and no
  scientific or novelty result is claimed.

## 2026-07-27 Preservation and merge boundary

- Current decision overlay after full residual and exact-head review:
  - the locally committed primary line is recoverable at `7dec564d`, with the
    immutable 38-path evidence/source commit at `186590c8` and a verified
    full-history recovery bundle; it is not yet public or merged;
  - the independently validated private V2 inventory binds 2,584
    ignored/untracked files, 1,678,118,521 bytes, ten worktrees, two stashes,
    zero errors, and `cleanup_authorized: false`;
  - exact stash archive refs, rejected-candidate archive refs, incremental/full
    Git bundles, raw-evidence ZIPs, a primary/H2 evidence ZIP containing 140
    source payload files plus one internal manifest, and a one-entry
    unique-source ZIP close the immediate machine-loss gaps without treating
    any residual as disposable;
  - PR #292 `835f6199` is rejected at Tier B 70/Critical. Replacement
    `f2e41d42` now supplies 113/113 metric-lineage coverage, preserves all 103
    historical JSON/PNG bytes, and passes 28 hostile validator mutations, but
    remains a candidate. Second independent exact-candidate review approved its
    content at local Tier B 100, making it content-ready for owner-approved
    publication/CI. Current public-state BHS is still 70/Critical because #292
    remains at `eb750958`, its body exposes stale invalid claims, and no hosted
    run exists for `f2e41d42`;
  - PR #293 `454e4a32` is rejected at 70/Critical after six iterations and must
    be withdrawn; no replacement should be opened absent an authorized real
    consumer;
  - PR #294 `6e78cf41` reached 100 only on its original stacked/pre-transplant
    head; independence analysis supports a new transplant branch/replacement
    PR containing its own commits without replaying rejected #293, followed by
    fresh exact-head review. Do not force-push or retarget public #294;
  - PR #295 `730b305e` is rejected at 60/Critical after five iterations; the
    current PR must be withdrawn. The reusable promotion plane, runner, engine
    integration, and runtime claims are not mergeable.
    The smallest defensible archive is eight independently hash-verified
    preregistration/manifest/lock/marker/reconciliation blobs plus one newly
    written narrowed disposition document in a separate replacement PR, based
    directly on final #292 with no #295 commit cherry-picked.
- No public push, force update, PR edit/close/open, merge, reset, restoration,
  worktree removal, stash drop, commit-graph rewrite, GC, prune, or cleanup has
  been performed. Explicit approval covers each public mutation, not merely
  pushes. Cleanup remains a later item-by-item owner decision.
- The immediate risk is durability, not scientific execution. The exact
  38-path primary RB-10/RB-11 source, test, protocol, and bounded-evidence
  allowlist is now preserved in local commit
  `186590c8bde8311f39c17167110d5ba30b13a4fd`; remote publication and equality
  proof remain.
- The primary branch is not the whole preservation surface. Git initially
  reported five additional linked worktrees on distinct branches, and four
  dedicated PR-repair worktrees were later added. The primary plus those nine
  linked worktrees make ten total. All are retained and classified; unique
  branch publication and external residual manifests remain separate work.
- A clean primary status after a future commit will not prove that worktree-only
  research or test data is safe. Completion requires both remote commit
  reachability and an explicit residual ledger.
- Cleanup is a separate authorization domain. Until the merged state is
  verified, no worktree removal, ignored-file deletion, branch pruning,
  artifact regeneration, reset, or stash dropping is permitted.
- Repository workflow consequence: preservation commits are not considered
  safely merged merely because a branch is pushed. ARCH-AEP requires an
  explicit scope lock, one authoritative backlog/tracker, exact-head
  verification evidence mirrored in the cycle log, a cohesive PR, and a phase
  summary/closure record. The tracker pointer—not a guessed historical
  tracker—is authoritative.
- Human checkpoints are part of the documented workflow. The present user
  request authorizes the preservation/merge scope; it does not authorize the
  later cleanup checkpoint.
- The authoritative tracker pointer and indexes now select the open
  `AEP-20260727-7` preservation cycle, with its scope lock, tracker, backlog,
  and verification log. The already-closed `AEP-2026-05-01` Model-Scope cycle
  remains historical and must not be reused.
- The recovered primary worktree initially had exactly 23 untracked upload
  candidates. The final primary allowlist grew to 38 paths after the cycle
  records and preservation plan were added, and all 38 are now in
  `186590c8`. Git also reported 1,654 ignored paths, including recovery bundles/zips,
  retired-branch notes, caches, linked worktrees, and experiment outputs.
  Ignore status is not a disposition: recovery and experiment surfaces require
  content/containment review, while caches and bytecode are expected
  reproducible candidates.
- Two stashes exist and are part of the residual ledger:
  `feat/brain-file-map-b0-b1: unrelated-wip` (2026-06-29) and
  `feat/a4-swap-campaign` WIP (2026-06-21). Neither may be dropped or assumed
  merged without inspecting its base, patch, untracked payload, and commit
  containment.
- The pre-reconciliation primary 13-commit stack at `7dec564d` has a real
  upstream dependency: its first four commits are exactly the head of open PR
  #292 (`lattice/phase2-continue-20260713` at `eb750958`). The remaining nine
  commits are the unpublished semantic-cache, CRSV, prime-ring, evidence, and
  first bookkeeping work.
  Duplicating #292 inside a new independent PR would obscure review and merge
  history; the merge plan must land #292 first or deliberately stack the new PR
  on its exact head.
- The apparent 957,566-line deletion in the primary comparison is almost
  entirely PR #292's deliberate untracking of raw NFCorpus run JSON/log data.
  This is not an unexplained loss: the deleted blobs remain in Git history,
  PR #292 is already a durable GitHub ref, and the local recovery ZIP contains
  the per-run data. Nevertheless, no local copy or recovery archive will be
  removed in this cycle.
- Public PR #292 remains at `eb750958`; its old public checks do not validate
  either local recondition candidate. Candidate `835f6199` was rejected at
  Tier B 70/Critical because the sidecar omitted 102 of 113 affected artifacts,
  failed to bind original versus annotated document blobs, and used a
  fail-open validator. Superseding local candidate `f2e41d42` meets the
  preservation, coverage, and hostile-test requirements, but it remains
  unpublished; its public branch/body, hosted checks, and fresh
  exact-public-head review must still close at 100 before merge.
- Recovery branch size is GitHub-compatible by individual blob size even when
  its bundle is not: the drift recovery commit's largest blob is about
  16.1 MB, while the bundle itself is about 120.8 MB and cannot be committed as
  an ordinary GitHub blob. Publishing the branch ref is therefore preferable
  to uploading the bundle.
- The ignored waypoint corpus is not an orphaned directory: the verified
  recovery bundle contains a complete branch at `65ae99a4`, and the recovery
  index reports 201 original files/62,042,044 bytes matched byte-for-byte.
  Remote branch publication can preserve that corpus without force-adding the
  locally excluded duplicate tree.
- Primary upload audit result:
  - all four RB-10 JSON SHA-256 values and internal manifest/artifact digests
    still recompute exactly;
  - the JSONs total only about 28 KB and need no Git LFS;
  - no credential/private-key pattern was found in the reviewed upload
    candidates;
  - the one changed planning-log user-profile path was redacted to
    `%USERPROFILE%`, and an added-line-only staged scan found no
    literal user-profile path;
  - repository `LICENSE` says Apache-2.0 while `pyproject.toml` declares MIT;
    all 26 branch-added root modules are now registered and verified through an
    isolated built-wheel import, but the license ambiguity remains an owner/
    legal decision;
  - the retained JSON schema does not bind source commit, command,
    interpreter/dependencies/OS, or raw-file SHA. The companion preservation
    ledger now binds its exact raw hashes and commit `186590c8`; it does not
    invent missing historical producer metadata or rewrite the evidence.

## 2026-07-26 Code-Backed Closure Audit

- This section is the current decision overlay. It preserves the dated RB-9 and
  RB-10 records below but supersedes any wording that called the frozen
  50/50 result a validated production-decoder theorem or described Stage 4 as
  unconditionally queued.
- Live validation:
  - 69/69 focused leading-shell, JO1, A1, G2, RB-10 contract, and runner tests
    passed in 9.017 seconds;
  - the retained RB-10 manifest returned `True` when rebound to every current
    artifact filename, digest, byte count, stage ID, and resource field;
  - an independent mesh audit reran 24/24 focused tests and independently
    checked all 38,610 stack-versus-flat distance comparisons;
  - the four JSON evidence files were not regenerated or edited.
- The decoder review found a real scope failure hidden by the former
  "implemented decoder closed" wording. `prime_ring_leading_shell.py` derives
  shell/tail formulas, sets the type-one inclusive ratio analytically, and
  forms the 50/50 result by averaging strict and inclusive leading terms. Its
  tests call the final tie helper on `[1.0, 1.0]`; they do not feed boundary
  observations through the float-FFT dense decoder.
- A bounded production-path audit constructed midpoint observations between a
  representative truth and every one of its 56 leading wrong-type competitors,
  for both truth types and `p in {11,19,31}`:

  | p | truth type | exact/direct ties | float-FFT exact ties | FFT winner type 0 | FFT winner type 1 | largest absolute FFT margin |
  |---:|---:|---:|---:|---:|---:|---:|
  | 11 | 0 | 56/56 | 27/56 | 30 | 26 | `1.1102230246251565e-16` |
  | 11 | 1 | 56/56 | 32/56 | 51 | 5 | `1.1102230246251565e-16` |
  | 19 | 0 | 56/56 | 17/56 | 19 | 37 | `2.220446049250313e-16` |
  | 19 | 1 | 56/56 | 7/56 | 8 | 48 | `3.3306690738754696e-16` |
  | 31 | 0 | 56/56 | 15/56 | 22 | 34 | `1.1102230246251565e-16` |
  | 31 | 1 | 56/56 | 19/56 | 56 | 0 | `1.1102230246251565e-16` |

  Direct dot-product scoring agreed with exact integer Hamming distance on all
  `336/336` ties. The float-FFT path preserved only `117/336` as bit-exact
  ties. It uses bit-exact floating equality, so roundoff turns many
  mathematical ties into strict numerical wins and can reverse the declared
  canonical winner. Disposition:
  - `PRW-T1R-TE-EVENT-UNION = CLOSED_INTERNAL_ANALYTIC_SCOPE`;
  - `PRW-T1D-ALL-STATES = ANALYTIC_TIE_COROLLARY_COMPLETE`;
  - `CURRENT_FLOAT_FFT_PRODUCTION_LINKAGE = FAILED`;
  - finite class-error validation and independent proof review remain open.
- `PRW-JO1` has two distinct dispositions that must not be conflated:
  - stacking/dimensional mechanism: closed by the exact identity between a
    fixed stack and its flattened longer code;
  - constrained ordinary code design: conditional and likely negative, because
    15 schedules tie and complementary/random controls match, but external
    known-design and unrestricted matched-work controls remain unrun.
- `PRW-A1` must not be killed from radial multiset equivalence alone. For every
  audited truth the ten actions have distinct fixed-label fingerprints and all
  45 action pairs have crossovers. The current result establishes truth-local
  competitor permutations, not one global prior-preserving state permutation
  or common channel isometry. The next exact gate is full action-channel
  conjugacy plus real semantic action availability and cost. A policy campaign
  is authorized only if that survives.
- `PRW-G2` validates necessary interaction existence but not an advantage:
  pair, strict three-way, and query-interaction residuals survive at `p=7` and
  bounded `p=11` unit checks; exact flat and factorized decisions agree and
  `static_control_advantage_established` is false. `PRW-G2H` remains narrowly
  open because the recorded pairwise auxiliary reduction is only an upper
  bound (`minimal_auxiliary_cost_proved=false`). Stage 4 requires an
  auxiliary-minimality result and a concrete orbit-specific
  representation/decoder frontier first. `PRW-G2A` remains blocked.
- The primary Nguyen--Györfi--Massey paper was checked directly. It represents
  `GF(p)` symbols by cyclic shifts of a binary `p`-tuple, proves the binary
  distance is the outer-code distance times the inner representation distance,
  and gives a Legendre representation with distance `(p+1)/2`. This is a
  direct construction collision for the unmasked PRW distance identity. It
  does not by itself settle the seven-shell/five-overlap event-union theorem,
  so publication novelty remains a specialist proof-equivalence question.
- Durability correction: the manifest and artifacts are content-valid on the
  current tree, but all four JSONs, the RB-10 code/tests, and three research
  documents are untracked. They are local evidence, not fresh-checkout or
  published evidence.
- Current execution order is therefore:
  0. preserve the exact tree;
  1. resolve the production numerical-tie contract;
  2. independently audit the T1R proof/construction equivalence if publication
     is pursued;
  3. run the A1 global conjugacy/action-cost gate;
  4. run the G2H auxiliary-minimality/named-candidate gate;
  5. optionally close the ordinary JO1 code controls;
  6. run noisy/nonlinear quotient, matched transcript, common-channel carrier,
     RSS/checkpoint fault injection, and bounded flat real retrieval.

## 2026-07-25 RB-10 Bounded Execution Findings

- Historical execution record. The 2026-07-26 code-backed audit above
  supersedes its decoder, durability, JO1, A1, and Stage-4 status wording.
- The prerequisite gates are now closed at their declared METHOD_DEV scope:
  `PRW-T1D-ALL-STATES` is proved for the frozen 50/50 type mixture, 14/14
  focused decoder tests pass, and the primary-source `PRW-T1R` claim chart is
  complete but leaves novelty unresolved.
- Three exact cells ran sequentially in fresh Python child processes. The
  content-rebound manifest is
  `artifacts/method-dev/prime-ring/rb10-bounded-experiment-manifest.json`.
  It reopens each artifact and checks its filename, digest, COMPLETE status,
  byte length, modeled resource fields, and stage ID.
- Memory was not remotely close to the 256 MiB reactive threshold:
  - `PRW-JO1`: 26,767,360-byte peak working set, 0.469 seconds;
  - `PRW-A1`: 26,345,472-byte peak working set, 0.312 seconds;
  - `PRW-G2-p7`: 26,087,424-byte peak working set, 0.156 seconds.
  The runner had approximately 23.44 GB available before execution and
  requires at least 1 GiB before each child. The 10 ms RSS monitor is reactive,
  not a hard kernel allocation cap; the safe conclusion applies to these tiny,
  trusted, sequential cells, not arbitrary future experiments.
- `PRW-JO1` produced a real but narrow shell-shaping fact. Schedule `(1,5)`
  keeps minimum distance `76` and reduces its multiplicity from `56` to `19`;
  its exact pairwise-tail union-bound surrogate is `0.637045` of aligned
  repetition. This is not a unique construction result: 15 of 55 restricted
  schedules tie, and the preregistered complementary `(1,10)` and seeded-random
  `(3,8)` controls have the same spectrum. The stacked and flattened codewords
  agree for every competitor. Known-design and unrestricted matched-cost
  controls remain incomplete. Disposition:
  `RESTRICTED_SHELL_SHAPING_LEAD_CONTROLS_INCOMPLETE`, non-promotional.
- `PRW-A1` passes only a fixed-label action-nondegeneracy gate. The ten actions
  have ten ordered semantic fingerprints, 160 of 351 competitors change
  distance across actions, and all 45 action pairs contain a ranking
  crossover. However, every action has the same sorted distance multiset.
  Thus the apparent difference is a semantic competitor permutation unless
  the labels, posterior, and action availability give it operational meaning.
  No posterior policy, MaxEJS comparison, acquired-bit saving, selector saving,
  or semantic-relabeling cost has been tested.
- `PRW-G2-p7` confirms ordinary irreducible factor behavior, not an advantage.
  The pair factor has a nonzero anchored mixed residual, the modular-sum
  hyperfactor has a nonzero three-way Möbius residual, the ambiguous path has
  four diagonal solution orbits rather than one fixed-offset orbit, and the
  query can change an interaction rather than only a unary term. Exact flat
  and factorized scores and selected MAP assignment agree. Independent,
  shuffled-label, wrong-grouping, and deterministic matched-random controls
  all satisfy their frozen counts; static advantage remains false. This
  licenses only an information-fair noisy recovery discriminator at Stage 4.
  It does not license `PRW-G2A`, approximate-message-passing, a real mesh, or a
  novelty claim.
- A fresh adversarial reviewer returned GO for the bounded execution after
  contract fixes, with no remaining P0/P1. The remaining P2 is recovery-only:
  a rare failure after all children finish but before the success manifest is
  persisted can leave completed artifacts without a recovery manifest. This
  fails closed because no COMPLETE manifest exists and requires a fresh output
  directory.
- Current bottom line: the execution found useful exact discriminators and a
  restricted shell-shaping effect, but no groundbreaking or novel computing
  mechanism. The most informative remaining work is to decide whether the
  ordinary matched controls close `PRW-JO1`, whether semantic action costs make
  `PRW-A1` operationally nontrivial, and whether `PRW-G2` improves noisy joint
  recovery after identical information is granted to a flat comparator.

## 2026-07-25 RB-10 Orbit-Plank and Irreducible-Mesh Planning

- Superseded as current status by the bounded execution findings above; retained
  as the preregistration history.
- The new work is now separated into four falsifiable lanes:
  - `PRW-JO1`: fixed joint-plank orbit-spectrum coding;
  - `PRW-A1`: posterior-guided orbit/plank acquisition;
  - `PRW-G2`: irreducible orbit-coded pair and hypergraph factors; and
  - `PRW-G2A`: adaptive factor acquisition, conditional on a static factor
    surviving.
- Fixed planks concatenate into one longer code. The flattened identical
  codebook is therefore a mandatory equivalence control; a gain over a weaker
  single-plank baseline cannot support a new mechanism claim.
- Adaptive next-plank selection is an instance of controlled sensing/active
  sequential hypothesis testing. Chernoff, MaxEJS, mutual-information-greedy,
  static, random, incremental-redundancy, and stop-only policies are mandatory
  controls. The only residual candidate is an orbit-restricted action family
  with nondegenerate observation laws and a new coverage, speed, or matched-cost
  result.
- `PRW-G1` remains closed for connected, cycle-consistent fixed-difference
  graphs. `PRW-G2` admits only pair factors with nonzero double-centered
  interaction, genuine higher-order residuals, query-dependent interaction
  changes, or explicitly noisy/frustrated/multiple-latent variants.
- The queue is proof-first and resource-bounded: close the type-one/balanced
  decoder, finish the `PRW-T1R` primary-source claim chart, establish a common
  channel, test `PRW-JO1`, conditionally test `PRW-A1`, algebraically screen
  `PRW-G2`, then conditionally test static and adaptive meshes. Generic learning,
  real retrieval, and broader scaling remain later gates.
- The first mesh slice is limited to `p in {7,11}`, 3--6 nodes, arity at most
  3, at most 8 factors, at most 200,000 exact assignments, 256 MiB modeled peak,
  25,000,000 modeled work units, and 30 seconds per cell.
- The canonical preregistration is
  `docs/research/prime-ring-orbit-plank-mesh-protocol-2026-07.md`.
- This was a planning and hypothesis-formalization pass only. No new
  implementation or experiment was run, and no novelty, retrieval, training,
  cost, speed, or production claim is supported.

## 2026-07-25 RB-9 Reconditioned Theorem and Learned-Cost Screens

- The original nearest-only `PRW-T1` event-union normalization is false in its
  declared tie-as-error scope. For fixed \(q\in(0,1/2)\),
  \(d_p=(7p-1)/2\), and the frozen two-type/eight-layer `typed16` bank, the
  eight wrong-type states at \(d_p+4\) contribute a nonvanishing fraction:
  \[
  \frac{8\beta_q(d_p+4)}{56\beta_q(d_p)}
  \longrightarrow \frac{[4q(1-q)]^2}{7}.
  \]
  The old normalized event union therefore tends to
  \(1+[4q(1-q)]^2/7\), not one.
- The repaired leading term is
  \(B_p(q)=56\beta_q(d_p)+8\beta_q(d_p+4)\). The exact seven-shell distance
  spectrum has a linear gap after those 64 states. The five exact leading-pair
  classes cover all 2,016 pairs, and each joint event has a strictly larger
  large-deviation rate than a single leading event. Bonferroni therefore gives
  \[
  \Pr\!\left(\bigcup_i E_i^{\ge}\right)=B_p(q)(1+o(1))
  \]
  for every transmitted state's wrong-type tie-or-better competitor-event union
  in the frozen bank. Status is
  `PRW-T1R-TE-EVENT-UNION: PROVED_ASYMPTOTIC`. The formal proof note now derives
  all seven shell counts, all five overlap-class counts, and the regular
  Hamming-isometry action transferring the theorem across all `32p` states.
- At \(p=4691,q=0.20\), the direct stable-tail adjacent/nearest ratio is
  `0.0585071601`, versus limiting correction `0.0585142857`; the other five
  shells contribute approximately `5.55e-225` relative to the repaired leading
  term. This is a numerical cross-check, not the proof. Convergence is not
  uniform near \(q=1/2\): at the same prime the nonleading/leading ratio is
  approximately `0.00834` for `q=0.45` but about `693` for `q=0.49`.
- The production decoder's exact tie rule is now bound for canonical type-zero
  truth: lowest canonical type ID wins, so wrong type one must strictly beat
  the transmitted state. With
  \(S_p=56\Pr[\operatorname{Bin}(d_p,q)>d_p/2]+
  8\Pr[\operatorname{Bin}(d_p+4,q)>(d_p+4)/2]\), canonical class error is
  \(S_p(1+o(1))\) and \(S_p/B_p\to q/(1-q)\). Status is
  `PRW-T1D-CANONICAL-FIRST-TYPE0:
  PROVED_ASYMPTOTIC_WITHIN_FROZEN_CANONICAL_STATE`. Type-one truth receives the
  opposite tie treatment, so the balanced/all-state theorem remains open.
- The raw-coordinate learned quotient screen recovers coefficients
  `(6,1,2,5)` with zero coefficient sum modulo seven and bias `3` across seeds
  `{7,42,1337}`. It is exact on unseen gauges and all gauges of 69 withheld
  quotient classes; the gauge-sensitive anchored control has orbit
  disagreement `1.0`. Four independent raw rows are rank four and the fifth
  reaches rank five and exactly identifies the rule for every seed. The status
  `NARROW_EXACT_AFFINE_SOLVER_RECOVERY_NON_CONFIRMATORY` is deliberately narrow:
  this is algebraic solver verification because both teacher and learner are
  noiseless modular affine, not evidence of empirical emergence.
- The paid-transcript screen finds a candidate structural-prior/training-search
  signal when the factorized learner is given the planted causal groups for
  XOR/parity teachers under spurious reversal. The unrestricted flat learner's
  shifted failure comes from choosing a simpler direct-payload shortcut under
  its MDL tie-break, not from inadequate capacity. A causal-only flat
  counterfactual also reaches shifted route accuracy `1.0`; its modeled search
  work is `9.0x` and `118.2x` the factorized work in the two frozen nonlinear
  cells. An equal-complexity wrong-group factorization underperforms, confirming
  that the advantage comes from the supplied grouping rather than the
  factorized form alone. True no-shift and split-hashed nuisance-remapping nulls
  are now distinct, and the result retains a hashed run contract covering
  teacher cells, thresholds, learners, train sizes, and resource limits.
  Factorized and flat controls have the same padded storage and inference
  envelope, and the paid-transcript control closes capacity once the bits are
  supplied. Its reconditioned status is
  `CANDIDATE_STRUCTURAL_PRIOR_TRAINING_SEARCH_SIGNAL_NON_CONFIRMATORY`.
- Validation is green for the bounded slice: 45/45 new tests and 362/362
  related prime-ring/CRSV regressions pass; scoped Ruff, Ruff format, and Python
  3.9 AST parsing pass for all six new files. Whole-repository discovery,
  measured RSS, checkpoint/resume, neural learning, real-corpus retrieval,
  utility, and novelty are not validated by this result.
- The highest-information next tests are, in order: final decoder tie policy;
  a fresh primary-source `PRW-T1R` claim chart; noisy/nonlinear generic quotient
  learning; matched generic transcript inference; a common-channel
  carrier/control screen; a bounded real-corpus pilot; and process-tree
  RSS/checkpoint fault injection. Queue-conditioned or “living” state remains
  blocked until real retrieval survives.

## 2026-07-24 RB-8 Full-Box Preflight
- Available memory is not the limiting factor for the next exact mathematics:
  `p=11` and `p=19` `PRW-T1` analyses are admitted by both byte and work
  ceilings. The present `p=31` formulation is rejected for work, not memory:
  152,591,424 estimated work units versus the immutable 50,000,000 ceiling.
  Raising the ceiling would hide the duplicated pairwise work; a grouped or
  symmetry-reduced formulation is the appropriate reconditioning.
- The large-prime Rader tests are cheap enough to run directly after focused
  correctness checks. Their conservative benchmark estimates are about 1.03
  MiB (`p=4091`) and 1.18 MiB (`p=4691`), so they are timing questions rather
  than memory-risk questions.
- The full METHOD_DEV campaign is qualitatively different from the bounded
  smoke: 11,520 retained cells versus 576, with primary SELECT/REPORT budgets
  of 512/1,024 rather than 2-8. The implementation validates factor shapes but
  exposes no campaign-wide work or wall-time estimator. A timed smoke or
  reduced lower-rate slice is required before an honest full-run decision.
- The formal quotient screen validates a precise but narrower claim than the
  original language: for the declared global-addition action on
  `Z_7^4`, an observer that factors through relative coordinates has 343
  distinguishable behaviors, rank 3, and exactly one gauge-null direction.
  The anchored observer separates every nonzero gauge move. This demonstrates
  how apparent ambient dimension can be a coordinate redundancy; it does not
  establish that higher-dimensional structure generally is a projection
  artifact.
- Runtime conditioning does not evade that accounting by itself. For arbitrary
  explicit Boolean predicates over the seven-bit current state and two-bit
  query, every accepted four-operation/two-gate program compiles exactly to at
  most four affine-plus-last-write-wins leaves. Once the flat comparator is
  granted the paid branch transcript, no extra behavioral state remains. Any
  surviving claim must therefore come from learning/compression/cost, a
  dynamically computed rewrite, or information not granted to the comparator.
- The held-out XOR experiment makes that distinction concrete. Conditional
  routing genuinely improves over payload-only and additive controls, including
  perfect correction of their planted misses, but an exactly transcript-matched
  four-leaf flat table reproduces every held-out decision. The useful object is
  the learned two-bit transcript, not an additional hidden state after those
  bits exist. The next defensible hypothesis is therefore about inference or
  compression cost for the transcript, not superior representational capacity.
  The synthetic `p=7/11` vector lengths do not test prime-ring arithmetic. Its
  held-out split is example-ID disjoint only; it does not establish
  distribution-shift or class-separation generalization.
- The three-point `PRW-T1` scale result splits in strength. Nearest-event
  clustering drops sharply from `0.256219` at `p=11`, to `0.017856` at `p=19`,
  to `0.000374874` at `p=31`. That is the strongest surviving mathematical
  signal in the current work. Farther-neighbor union mass, however, is
  `0.360077`, `0.403705`, and `0.240863` relative to the nearest term: still
  order-one and non-monotone. Since both terms must vanish, three finite points
  do not establish the conjecture. Deterministic distinct-signature controls
  are exactly equal at every prime, so the visible result is not special to the
  ordered `0..7` phase signature.
- Exhausting all 352 transmitted type/shift/mask states at `p=11` produced one
  and only one digest for the label-free numeric/spectral/bound fields. Nearest
  state IDs transform equivariantly rather than remaining literally equal, and
  elapsed fields are nondeterministic. This closes empirical state averaging
  for that finite bank without overstating byte-identical result dictionaries.
- The corresponding algebra is now exact for the declared two-type/RM(1,3)
  bank. Additive phase shifts, RM-mask multiplication, and a fixed type-swap
  involution act regularly on all `32p` transmitted states at `p=11,19,31`;
  therefore the distance vector, intersection spectrum, BSC terms, ordinary
  union bound, and Hunter weight are state-invariant. This removes a finite
  averaging ambiguity, but it neither proves asymptotic decay nor applies to
  arbitrary phase signatures, decoder error, retrieval utility, or novelty.
  The hardened `p=31` preflight counts 4,674,304 exact generator comparisons,
  15,309,808 conservative total work units, and a 3,822,464-byte co-resident
  peak model; all resource refusals occur before exact certificate work.
- The `p=31` run did not require weakening general resource safety. The
  analyzer now distinguishes 9,190,584 scalar/Python work units from
  61,011,968 vectorized int32 multiply-adds, retains the original 50-million
  scalar cap, and applies a narrow 64-million GEMM cap. This is operation
  accounting, not elapsed-time or process-RSS calibration.
- A more careful Rader probe reverses the tempting preliminary story. Warmed,
  alternating-order blocks put `p=4691` at a median `1.02805x` NumPy time, not
  a speedup; only 3/30 blocks meet a 5% improvement. `p=4091` is about `2.16x`
  NumPy time. The smooth `4690` factorization may make Rader competitive, but
  this implementation does not currently deliver a reproducible advantage.
- The final matched harness resolves the systems-relevant version more clearly.
  `p=4691` Rader DFT is locally competitive (`0.9619x` median NumPy time), but
  its cyclic-correlation path is `1.0398x` and wins only 7/18 blocks.
  `p=4091` correlation is `2.9393x` NumPy time and wins 0/18. Since retrieval
  needs correlation rather than an isolated forward transform, the current
  implementation fails RADER-1 even though the smooth 4690 convolution length
  remains an engineering hint.
- The exact-shape campaign profiler gives a useful scheduling estimate but not
  a safety proof: 300 trial groups took 19.494 seconds. The targeted raw-sanity
  slice retains 9,549 observations, while the full grid retains 21,165 and at
  least roughly 110 MB of artifact data by the independent audit. The targeted
  slice is the next admissible campaign only after global limits and atomic
  output exist.
- Campaign safety now changes that decision boundary. The exact targeted
  preflight is allowed at 229,502,456 modeled bytes and 12,648 retained rows;
  the broad grid is refused at 1,059,556,800 modeled bytes. The targeted run may
  now execute under deadline, row-cap, and atomic-write controls, while lack of
  measured RSS and checkpoint/resume remains explicit.
- The final hardened targeted campaign completed in 424.034 seconds. All 12
  preregistered `q in {0.00,0.20,0.35,0.45}` by seed
  `{7,42,1337}` cells completed and passed the raw-type sanity contract.
  The primary pooled report has recall `1.0`, zero false unlocks over 3,072
  unrelated report groups, and a one-sided 97.5% Clopper-Pearson upper bound
  of `0.00120009`. The equal-channel-use repeated-bit control reaches
  `0.999349` recall; native sparse OPPW reaches `0.908203` recall and uses a
  different symbol-substitution noise law, so a three-way superiority claim
  remains invalid. The campaign correctly leaves `PRW-1` through `PRW-4`
  inconclusive, promotion false, and novelty not established.
- The final campaign artifact is 1,897,602 bytes with SHA-256
  `fce8901e442bd4e1abece24e875d177db85b7c05bb330f8f13743135ee0755d1`.
  The highest periodic external poll observed 173,236,224 working-set bytes,
  but that is not a continuous process-peak measurement. The executable guard
  did not measure or enforce process RSS, so its own evidence correctly labels
  RSS unmeasured. Checkpoint/resume also remains explicitly unimplemented.
- The separate Rader artifact is 24,896 bytes with SHA-256
  `ae4d2af41070e9fa67f8428cfcec5284b92da53af6e62823a1fce9ca2807b0a2`.
  The campaign now says `NOT_TESTED_BY_THIS_CAMPAIGN`, records that the Rader
  implementation exists, and explicitly says the separate harness was not
  ingested. The separate harness remains the timing evidence and rejects the
  current correlation-speed hypothesis without making a general hardware
  claim.

## 2026-07-24 RB-7 Preregistration and Resource Screen
- The next bounded Fourier discriminator is `PRW-H1Q`: at `p=11`, compare every
  matched two-bin subset over a declared quantizer-origin grid, then report the
  complete rank distribution, origin spread, and an origin-averaged diagnostic.
  The evaluation may not choose a favorable subset or origin after seeing the
  outcomes. A finite origin average is a grid-robustness diagnostic, not a
  claim of stochastic dither unless the dither distribution and seed are
  independently fixed.
- There are five nonconjugate bins at `p=11`, hence ten two-bin subsets.
  Four quantized origins plus one unquantized baseline produce 50 candidate
  cells before controls. Calling the existing exact comparator independently
  for all 50 would duplicate the same 2,048 BSC patterns and is not approved
  until an aggregate bytes/work/deadline preflight passes. A shared streamed
  enumeration is the preferred formulation.
- A read-only estimator probe makes that refusal concrete. Each independent
  cell estimates only 2,408 model-level peak bytes, but 2,820,096 work units,
  already above the frozen 2,000,000 default. The naïve 50-cell grid totals
  141,004,800 work units, above even the 50,000,000 hard ceiling. The bank
  itself is tiny (541 estimated peak bytes and 44 work units), so memory is not
  the problem; duplicated decoding work is. No exact pattern loop ran during
  this estimate.
- A dependency-free group-action check found that the ten unordered `p=11`
  two-bin subsets form two multiplicative-relabeling orbits of five:
  `{(1,2),(1,5),(2,4),(3,4),(3,5)}` and
  `{(1,3),(1,4),(2,3),(2,5),(4,5)}`. The batch should report within-orbit
  spread as a symmetry audit. This does not assume phase quantization preserves
  the unquantized relabeling symmetry; breaking it by origin is itself a useful
  artifact diagnostic.
- Hostile analysis shows why a common four-origin grid is not a
  symmetry-closed dither. A multiplier can map one selected frequency through a
  negative representative (complex conjugation) while leaving the other
  positive, which maps per-bin origins as `(alpha_1, alpha_2) ->
  (+/- alpha_1, +/- alpha_2)`. The common-origin diagonal cannot represent all
  such maps. The current upper-tie rounding rule can also break conjugation at
  exact half-step boundaries. Consequently, the four common origins test
  nuisance sensitivity only. A stronger invariant screen needs a declared
  per-bin origin product grid, explicit boundary-hit accounting, or continuous
  origin integration, all with a new aggregate resource preflight.
- The final cached shared-stream preflight estimates 54,432 model-level bytes
  and 17,951,240 work units for 2,048 patterns, 50 conditions, and 102,400
  cached condition evaluations under one 25-second deadline. It computes one
  template FFT, 2,048 query FFTs, and no public reference-decoder calls; focused
  tests match the reference scores, winners, ties, margins, and abstentions.
  The straightforward symmetry-closed `A4^2` extension would require 170
  conditions, 348,160 cached evaluations, and 59,734,280 work units, exceeding
  both the local 25,000,000 and upstream 50,000,000 hard ceilings, so that route
  is explicitly refused.
- The root-owned final exact run completed in 22.359 seconds beneath the
  unchanged deadline. Unquantized accuracy is constant to floating precision
  within each predicted orbit: orbit A mean `0.1202012759` with spread
  `4.16e-17`, orbit B mean `0.1231158929` with spread `6.94e-17`; A minus B is
  `-0.0029146170`. The between-orbit gap is ordinary ratio-class geometry, not
  a special pair or resonance.
- Across the four common quantizer origins, 30 of 45 pairwise orderings reverse
  and no pair strictly dominates every other pair at every origin. Maximum
  per-pair origin spread is `0.0048568741`; maximum absolute grid-average
  change from the corresponding unquantized value is `0.0019161556`.
  Origin `0.5` has up to `0.0306274866` boundary-near probability mass, while
  `0.0009287012` zero-spectrum mass is separated and correctly abstains.
  Therefore the bounded result supports within-orbit unquantized equivariance
  and quantizer-origin sensitivity, not selected-frequency advantage.
- The proof-first `PRW-C1A` screen is narrower than the earlier geometric
  language: on the nonzero field elements, write `x = g^n` for a primitive root
  `g`. Multiplication by `g^a` is then the additive shift
  `n -> n + a (mod p-1)`; CRT coordinates merely relabel that shift
  componentwise. For the mathematical Legendre character on nonzero elements,
  the same action depends only on exponent parity and therefore reduces to a
  polarity bit. The zero element is a separate fixed point.
- The repository carrier uses a deliberately bipolar convention with coordinate
  zero set to `+1`, whereas the number-theoretic character has value zero there.
  Therefore an odd-exponent pivot flips every nonzero carrier coordinate while
  leaving coordinate zero at `+1`: the exact full-array reduction is a polarity
  bit plus one fixed-coordinate exception, not a literal global sign.
- `PRW-C1A` can establish an exact redundancy or degeneracy result, but CRT
  factorization alone does not establish compression, faster retrieval, or a
  new matching primitive. Any surviving advantage must beat explicit flat
  permutation and polarity-only controls at matched search opportunity.
- Allowing one independently chosen shift in each CRT factor still gives only
  `2 * 5 * 7 * 67 = 4,690` tuples, in bijection with one ordinary exponent
  shift. It does not create an extra address dimension. Giving each of `L`
  layers an independent pivot does create `4690^L` states, but that expansion
  must be charged as `L * log2(4690)` control bits and a correspondingly larger
  search space; it is not free capacity from the factorization.
- A further proof-first reduction covers pure onion layering. Additive
  rotations `T_b` and multiplicative pivots `M_a` generate the ordinary affine
  group, with `M_a T_b M_a^-1 = T_(a*b)`. Every word containing only those
  coordinate permutations collapses to one map `x -> a*x+b`, so ordering more
  such layers cannot by itself create a new high-order memory. A mechanism can
  escape this reduction only by inserting a declared nonlinear or conditional
  replacement; that replacement must then be tested against balanced random
  and conjugated sector-mask controls rather than attributing its effect to
  CRT.
- The effective multiplicative orbit is payload-dependent. For the punctured
  Legendre character, the 2,345 quadratic-residue pivots form its stabilizer
  and the orbit has only two states. Under the repository's bipolar
  zero-coordinate convention those are still just the original array and the
  common nonzero-coordinate complement with zero fixed.
- For future replacement tests, CRT-axis shifts conjugate a sector replacement
  into the same replacement at a shifted sector label. Disjoint overwrites
  commute; overlapping overwrites can show ordinary last-write-wins effects.
  Any claimed layer-order benefit must therefore match mask size, mask overlap,
  condition bits, and random balanced/coset partitions, and must survive a
  change of primitive-root address convention.
- The exact `PRW-C1A` screen now passes a root-owned direct run in 0.144 seconds
  with 3,310,582 estimated Python-object bytes and 2,176,160 work units
  (neither is measured process RSS). The CRT tuple count and shared pivot count
  are both exactly 4,690; the factorization creates no additional states.
- Eight independently assigned layers have `4690^8 =
  234089935364620159344100000000` possible control tuples and require about
  97.563 control bits. Those tuples were counted, not enumerated, and no
  payload-level independence or distinguishability was established. The
  Legendre stabilizer/orbit is
  `2345/2`, and the affine rotation-plus-pivot family has the ordinary upper
  bound `4691 * 4690 = 22,000,790`.
- Four algebraic kill criteria trigger: flat exponent-permutation relabeling,
  no extra CRT states, Legendre parity plus the fixed-zero exception, and
  affine collapse of rotation/pivot words. The systems control does not
  trigger because learned-payload utility and implementation cost were not
  tested. This closes new-group/new-CRT/new-onion-order interpretations, not a
  possible learned-layout engineering benefit.
- The `PRW-C1B` p=7 conditional-replacement screen also ran directly within a
  99,072-byte/29,808-work estimate and one-second deadline. Its representative
  four-operation program reduces to `x -> 3*x+6` followed by one exact
  last-write-wins overwrite; the two effective masks each have size three and
  overlap once. All 128 binary payloads confirm the normal form. Disjoint
  overwrites commute; overlapping overwrites show only declared
  last-write-wins order. The seeded random control matches mask
  size/overlap and overwrite conjugacy, but does not claim full affine-program
  or cost/utility equivalence.
- This first replacement screen uses fixed address masks and constant
  replacement bits. It does not test predicates or replacement functions that
  depend on payload, query, or layer state, so it narrows only the static-mask
  lane and does not kill the broader runtime-conditional hypothesis.
- The final current-tree validation passes 16/16 quantizer-null tests, 23/23
  CRT/algebra tests, and 14/14 conditional-replacement tests; the three suites
  pass together at 53/53. The CRT suite includes hostile numeric-subclass and
  greater-than-64-bit input paths so arbitrary big integers are refused before
  modular arithmetic or work allocation. Scoped Ruff, read-only AST parsing,
  and whitespace validation are separate correctness checks, not scientific
  replication.
- Folding those 53 checks into the prior eight-module regression set gives
  230/230 passing tests in 31.561 seconds on the final current tree. This is a
  low-memory deterministic regression run; it does not repeat the exhaustive
  `p=11` grid or constitute powered evidence.
- The accepted `p=11` run is one exact tiny synthetic grid, not a statistical
  replicate, continuous-dither study, corpus run, campaign, or retained
  evidence artifact. The `p=4691` result above is an algebraic table/check, not
  a transform, retrieval, or timing benchmark.

## 2026-07-24 Resource-Bounded Return
- Live recheck before new edits found
  `codex/prime-ring-onion-method-dev` at
  `08e3c2aeeb5c645a3c95f08a14ac101b226f8c9d`, clean and eight commits ahead
  of `origin/main`. This is local branch state only; no publication claim is
  implied.
- The previous checkpoint already implemented RB-1 matched controls, RB-2
  finite intersection diagnostics, RB-3 correctness-first Rader transforms,
  and typed remaining hypotheses. It did not establish novelty, systems
  superiority, production value, or a scientific result.
- The highest-information safe additions are currently:
  1. fail-closed aggregation over every preregistered lower-rate raw-sanity cell
     rather than only the pooled `q=0.45` cell;
  2. a tiny graph-coupled cyclic-fiber test with a degree/edge-matched shuffled
     graph control;
  3. a tiny multi-frequency phase-signature test with matched energy and search
     opportunity controls.
- These additions can test whether apparent gains come from cross-node
  coherence or frequency diversity, rather than extra energy, extra
  hypotheses, or a favorable graph. They remain synthetic mechanism tests.
- Repository brutal-honesty rules require runtime evidence for completion
  claims. The current slice therefore distinguishes bounded unit/property
  evidence from a powered campaign or real retrieval result.
- The default WindowsApps PowerShell launcher again failed with access denied
  after restart. The stable bundled Node runtime is being used for bounded
  inspection and command execution; the failed launcher path will not be
  retried.
- The live file map contains no existing `prime_ring_graph_fibers.py` or
  `prime_ring_multifrequency.py`. The relevant reusable primitives are the
  validated carriers/rotations/correlations in `prime_ring_waypoint.py`, the
  immutable finite-analysis caps in `prime_ring_intersection.py`, and the
  matched-control aggregation in `run_prime_ring_waypoint_experiment.py`.
- The formal ledger already defines `PRW-G1` as finite-group synchronization on
  a graph and `PRW-H1` as normalized discrete-Fourier phase synchronization.
  New code must preserve those ordinary mathematical meanings and may not turn
  the visual "3D", "gravity", or "resonance" language into a physical claim.
- A fresh primary-source collision check substantially narrows both ideas:
  - Singer's 2011 angular-synchronization work already estimates node phases
    from noisy graph-relative offsets, including consistency over general
    compact groups.
  - Perry, Wein, Bandeira, and Moitra (2016) explicitly treat compact-group
    synchronization with multiple Fourier/representation channels.
  - Gao and Zhao (2019) explicitly call the construction
    "Multi-Frequency Phase Synchronization."
  - Current sparse-graph work also uses connection/edge rotations and spanning
    structures for angular synchronization.
- Therefore neither graph-coupled cyclic fibers nor multi-frequency phase
  synchronization is novel in isolation. A surviving contribution would have
  to be a narrower interaction—such as the exact OPPW/type-mask construction,
  a proved resource/error tradeoff, or a held-out retrieval/correction
  advantage under matched controls.
- New bounded analyzers should copy the established finite-analysis safety
  pattern: immutable hard ceilings, caller budgets that may only lower those
  ceilings, shape/work preflight before allocation, a monotonic deadline, and
  `unittest`-only deterministic oracles compatible with Python 3.9.
- The protocol deliberately uses only `q=0.45` for the pooled false-unlock
  threshold cell, while separately requiring 100% raw type recovery at every
  rate through `0.45`. The lower-rate fix must not create four primary
  thresholds or reuse `REPORT` outcomes for selection; it should be a separate
  construction/noise-model sanity aggregation over raw decoder outputs.
- `PRW-H1` has a stronger null than initially stated. For prime `p`, any one
  nonzero Fourier bin is invertible modulo `p` and already distinguishes every
  clean cyclic shift. Multiple bins cannot add noiseless address capacity; they
  can only add redundancy/robustness under corruption. For real signals, bins
  `k` and `p-k` are conjugates rather than independent observations.
- The new bounded Fourier harness therefore:
  - restricts selected bins to `1..(p-1)/2`;
  - searches exactly `type_count * p` shared type/shift states for every
    condition;
  - normalizes phase features to unit energy;
  - includes same-count random bins, repeated-single-bin, single-bin,
    magnitude-only, randomized-phase, and time-domain controls;
  - supports explicit 1-16 bit phase quantization, including an 8-bit check;
  - streams exact BSC patterns only for tiny state spaces.
- The first PRW-H1 suite passed 17 deterministic tests in 0.525 seconds. This
  proves bounded implementation properties only; no frequency-diversity
  advantage was established.
- A tiny exact `p=7`, one-type, one-node BSC sweep enumerated all 128 corruption
  patterns at `q={0,.20,.35,.45}` with only 928 estimated peak bytes and 83,584
  estimated work units per condition set:
  - unquantized bins `(1,2)` and the same-count random control `(1,3)` had
    identical exact accuracy at every rate;
  - two distinct bins beat a repeated copy of one bin by `0.26112`, `0.109243`,
    and `0.026715` probability at `q=.20`, `.35`, and `.45`;
  - this is evidence for generic redundancy in this toy channel, not for a
    special frequency selection, harmonic resonance, or new address capacity;
  - 8-bit quantization introduced small bin-set differences (at most about
    `0.00834` here), which is more plausibly a quantization/bin interaction than
    a selected-bin result and requires all-bin/all-seed controls before any
    interpretation;
  - time-domain correlation used a different coefficient budget, so its toy
    accuracy is diagnostic rather than a matched superiority comparison.
- The lower-rate closure implementation is structurally separate from threshold
  fitting. It selects exactly one retained completed cell per frozen
  `(seed,q)` at the canonical 4691/Legendre/8-layer/typed16/payload-0.25
  configuration, requires the expected raw REPORT group count and accuracy
  exactly `1.0`, rejects missing/duplicate IDs and promotion-enabled records,
  and feeds its fail-closed result into the dense and all-mandatory control
  gates. Hostile review then showed that this first verifier trusted mutable
  summaries: fabricated noncanonical IDs, coerced numeric types, near-matching
  rates, and REPORT-derived threshold metadata could still close the gate.
  Provenance hardening is therefore required before the implementation itself
  can be called fail closed. The powered campaign remains unexecuted, so no
  empirical gate is closed.
- Fresh hostile review confirmed the PRW-H1 DFT sign/roll law across 2,100
  noiseless cases, but found preflight, condition-cap, Python 3.9, tie-margin,
  and accounting/control defects. Reconditioning now:
  - refuses malformed shapes and caller-tightened byte/condition limits before
    control construction or content scans;
  - uses a Python 3.9-compatible population count;
  - forces tied winners to margin zero;
  - distinguishes logical coefficient slots, stored scalars, energy, and
    arithmetic matching;
  - exposes proposed/random-bin collisions; and
  - includes the previously missing all-nonconjugate-frequency control.
- The reconditioned PRW-H1 suite now passes 37/37 tests in 2.669 seconds with
  clean focused Ruff. Final hostile review reports no P0, P1, or P2 findings.
- The second PRW-H1 reconditioning also gives the whole multi-condition
  comparison one aggregate work/deadline budget, guards the exported random-bin
  helper before `np.arange`, renames "independent" bins to the accurate
  "nonconjugate" term, counts bank-build magnitude temporaries, and states that
  phase-bit tests still use float64/complex128 arithmetic with no packed
  storage or speed claim. The focused suite remains 21/21 green in 0.523
  seconds with clean Ruff.
- A stronger artifact-free PRW-H1 falsifier exhaustively covered all three
  two-bin subsets of the nonconjugate `p=7` bins, all 128 BSC patterns,
  `q={.20,.35,.45}`, and 8-bit quantizer-grid origins
  `{0,.25,.5,.75}`:
  - all three unquantized two-bin subsets have exactly the same accuracy at
    each rate;
  - the apparent 8-bit bin ranking changes when only the quantizer origin
    changes;
  - the largest within-subset origin spreads are `0.0032768`, `0.0076534555`,
    and `0.0083385070` at the three rates;
  - the proposed two-bin subset trails the three-bin all-nonconjugate
    diagnostic in 29 of 36 quantized cells, although that diagnostic has a
    larger coefficient budget and is not a matched superiority comparator;
  - each run is guarded at no more than 1,288 estimated model-level bytes and
    101,504 work units.
  This kills a selected-frequency or resonance interpretation for the toy
  result. The distinct-bin gain over repeated copies remains ordinary
  redundancy, and the small 8-bit differences are grid-origin artifacts or
  interactions until stronger evidence says otherwise.
- The bounded `PRW-G1` graph analyzer now includes that explicit
  one-global-phase-plus-fixed-offset comparator. In the helpful `p=7`
  triangle, its assignment, score, tie count, planted margin, and planted rank
  are exactly identical to the graph-coupled decoder; every graph-minus-global
  effect is zero. Fifteen tests, focused Ruff, and an independent exhaustive
  check of all 511 nonempty labeled simple graphs on three nodes pass. The
  earlier margin gain over independent/shuffled controls is fully explained by
  restricting the decoder to the fixed-offset global-phase family, not by a
  new graph-memory mechanism.
- Warm model-array allocation checks at the largest permitted PRW-H1 shape
  (`p=31`, 32 types, 12 nodes, 15 bins) now remain below their declared
  estimates: bank construction measured 532,903 bytes versus a 579,871-byte
  estimate, one decode measured 755,648 versus 790,448, and the eight-control
  comparison measured 826,096 versus 846,000. These are `tracemalloc`
  diagnostics for Python/NumPy allocations, not process RSS or a production
  memory benchmark.
- The lower-rate raw-sanity verifier now binds every accepted cell to a
  canonical cell ID and campaign/stream-contract digest, requires exact Python
  scalar types and exact frozen coordinates, enforces SELECT-only thresholds
  with zero REPORT rows consumed, and recomputes row counts, group/order
  digests, row digests, accuracy, and canonical trial truth from planted IID
  REPORT lineage. An independent adversarial probe that changed both the
  retained true and predicted types, then recomputed the row digest, initially
  exposed one remaining gap; canonical `_trial_truth` recomputation now rejects
  it with `raw_provenance_true_type_not_canonical`.
- A second hostile attack showed that those digests still self-attested: after
  rewriting a failed prediction, correctness, counts, accuracy, and row digest,
  an attacker could reconstruct every old seal from serialized fields and turn
  the gate into `COMPLETE_PASS`. The accepted design now creates one per-run
  authority before campaign traversal, uses a random 32-byte HMAC-SHA256 key to
  authenticate each cell/campaign/carrier/raw-provenance/count/accuracy tuple,
  and never serializes the authority ID, key, or seals.
- Replaying the exact reconstruction attack with an unkeyed digest now yields
  `INCOMPLETE_FAIL_CLOSED`,
  `live_source_seal_authentication_failed`, `contract_complete=False`, and
  `gate_pass=False`. A replacement authority, missing or misbound seals,
  mutated/missing carrier provenance, and a SELECT/REPORT tag collision also
  fail closed. Independent hostile re-review reports no P0, P1, or P2 under the
  explicit serialized-input-tampering boundary; arbitrary same-process code
  execution is not claimed as a protected boundary.
- The 12 seals are conservatively budgeted at 24,576 bytes and the authority at
  2,048 bytes, for 26,624 estimated live-attestation bytes. This estimate is
  scoped to the extra seal/authority objects, not process RSS, cells, or the
  serialized raw-provenance rows already retained for auditability.
- The final complete bounded runner suite passes 29/29 tests, and the combined
  eight-module research suite passes 177/177 tests in 28.233 seconds. Scoped
  Ruff, read-only AST parsing, and whitespace validation also pass. These use
  only tiny smoke fixtures, exact small-prime enumeration, and fabricated
  in-memory provenance records; they do not invoke `full_config()`, the
  11,520-cell campaign, 4091/4691 timing, real retrieval, or retained evidence
  generation. This validates gate behavior, not the raw-sanity scientific
  criterion itself.
- The best next low-risk discriminator is not more `p=7` parameter searching.
  It is a preregistered, resource-estimated `p=11` quantizer-origin/dither null
  followed by an algebraic `PRW-C1` redundancy screen. Neither may inherit
  support from the current negative graph/Fourier results.

## 2026-07-23 Prime-Ring Onion-Lattice Findings
- Live `origin/main` already contains thirteen lattice commits after the
  recovered brain-file-map branch:
  - H6 temperature schedule (`#280`);
  - H5 steering-post bank, post-bank construction/runtime/conditions, and
    head-to-head driver (`#279`, `#281`-`#283`, `#290`);
  - rollback-backed steering control and promotion routing (`#284`-`#286`);
  - H3 teacher-supervised correction and C3b wiring (`#287`-`#288`);
  - H4 compound-cycle control (`#291`).
- Those commits provide implementation surfaces and unit coverage, but the
  current audit has not yet found durable real H5 head-to-head wins or evidence
  that H4 compounding improves retrieval. Existing test prose explicitly says
  the H5 win question belongs to the campaign rather than the wiring test.
- The local ignored waypoint-research folder contains newer July planning,
  campaign evidence, paper drafts, novelty assessments, harmonic-invariance
  preregistration, and lattice endgame documents. These are research context,
  not automatically durable repository evidence.
- Canonical BCC-1 recovery is v9 on `codex/semantic-cache-coherence-h1` at
  `11d548bd`, with pack, manifest, result, protocol, prior-art manifest, runner,
  and 171-test recovery evidence. The v8 preformat directory is superseded and
  must not be used as the implementation base.
- The same branch contains later commit `2b0fa044`, whose subject says the H2
  reruns closed both `CD-H1-01` and `CD-A2-01`. Exact artifacts and flag
  synchronization still require live audit before accepting that disposition.
- July lattice endgame records state that the corrector question was answered
  negatively and the powered estimator path also failed: annealing, living,
  and bounded correctors matched or lost to trivial controls. This is a
  disconfirming result, not a foundation for the prime-ring hypothesis.
- The live H5 implementation explains at least one negative structurally:
  `C5` and static `C5s` construct the same deterministic bounded bank, route
  from the same pre-drift snapshot, and `C5` deterministically rebuilds pruned
  posts from the same anchors/seed before applying them. They should tie, while
  the existing strict gate requires `C5 > C5s`. That gate is unwinnable without
  a genuinely state-changing router or bank update.
- Hypothesis naming is already ambiguous: durable diagnostics use `H3` for the
  confirmed C2 oracle-re-embed result, while later PRs reuse `H3` for unrun C3b
  teacher distillation. New work must use a separate `PRW-*` registry and may
  not inherit any bare `H3` status.
- The merged H3-H6 additions are not complete campaign surfaces:
  - run-level correction-norm aggregation omits C3b and C5 variants;
  - bounded runtime C3b discards learning diagnostics and has no NDCG campaign;
  - H6 controls only post-bank pruning, not the roadmap's shared
    sedimentation/updater/ES schedule or high-temperature perturbation;
  - H4/H5 campaign knobs are absent from the CLI and default campaign;
  - their deferred GPU campaigns were not carried into `docs/next-session.md`.
- Despite their `feat(lattice)` labels, the merged H3-H6 code contains no
  mathematical lattice, DAG, sheaf, holonomy, ring, or 4691 mechanism. H5 is a
  flat centroid-to-correction dictionary. The prime/onion mechanism therefore
  has not been tested by those commits.
- `codex/crsv-onion-method-dev` at `59378814` is an unmerged local branch whose
  name and recovery placement indicate likely overlap with conditional
  replacement/subdomain variance and onion-layer work. Audit it before adding
  a second implementation.
- The block-flag script mechanically passes `CLEAR` but reports two open carried
  debts. Their June dates are older than the rulebook's one-cycle/five-day
  limit, so the declared flag may be stale. New feature implementation remains
  paused while exact debt disposition is reconstructed.
- After fast-forwarding to the recovered canonical baseline, commit `2b0fa044`
  supplies the real swap-path campaign evidence and closes both rows.
  `scripts/check_block_flag.py` now reports `CLEAR`, zero carried-debt rows, and
  `PASS`; isolated METHOD_DEV feature work is permitted.
- The campaign now runs on isolated branch
  `codex/prime-ring-onion-method-dev` at current `origin/main`; the user's
  `feat/brain-file-map-b0-b1` commit remains preserved on its original branch.
- `4691` is prime, `4691 mod 4 = 3`, and
  `4691 - 1 = 2 * 5 * 7 * 67`.
- The length-4691 bipolar Legendre sequence was checked exhaustively in the
  current session: zero-shift correlation is `4691`; every one of the 4,690
  nonzero additive shifts has correlation `-1`.
- `4091` is also prime and `3 mod 4`, has the same Legendre property, and is a
  mandatory hardware-friendly control.
- Post-restart full-size exact verification passed for both carriers:
  - `4091`: prime and all 4,090 nonzero autocorrelations equal `-1`;
  - `4691`: prime and all 4,690 nonzero autocorrelations equal `-1`;
  - `4690` factors exactly as `(2, 5, 7, 67)`;
  - `2` is a primitive root modulo `4691`.
- A full-size eight-layer quotient probe at `p=4691` produced:
  - correct shared-shift score `1.0`;
  - a wrong signature with at most one aligned layer scored
    `0.12481347260712003`;
  - observed margin `0.8751865273928799`;
  - conservative analytic wrong-score bound `0.12518652739287997`.
  This validates the noiseless separation construction, not semantic utility.
- There are two distinct singleton/binary ideas:
  - Rader separates `{0}` from the 4,690 nonzero field coordinates. This is one
    anchor scalar per channel, not inherently one bit.
  - `Z_4690 ~= Z_2 x Z_5 x Z_7 x Z_67`; the `Z_2` factor supplies one
    multiplicative polarity character per ring.
- Eight binary rings can therefore form an eight-bit cross-layer orientation
  mask with 256 configurations. Free search over all masks also creates a
  256-way false-match opportunity, so masks require type/provenance constraints
  or multiple-comparison-corrected thresholds.
- Additive phase recovery and multiplicative payload pivots are different:
  - additive shifts create the 4,691 ideal Legendre alignments;
  - multiplicative action on a pure Legendre carrier yields only the original
    pattern or its sign inverse;
  - CRT pivots must be tested on arbitrary/learned payloads.
- With primitive root `2`, a nonzero exponent has CRT coordinates
  `(n mod 2, n mod 5, n mod 7, n mod 67)`. Exact reconstruction is
  `n = 2345*n2 + 1876*n5 + 2010*n7 + 3150*n67 (mod 4690)`.
- Eight binary 4,691-bit planes occupy exactly 4,691 bytes only as a
  theoretical bit-packed representation. The current semantic payload API uses
  `float64`: one `8 x 4691` payload occupies 300,224 array bytes. Packed binary
  accounting cannot be applied to those real-valued payloads.
- The strongest existing-prior-art collision is Residue Hyperdimensional
  Computing plus resonator decoding. HRR/FHRR, Legendre/Paley codes,
  product-key memory, cellular sheaves, and low-rank adapters cover other
  individual components.
- The current defensible research target is the integration and causal
  interaction of typed lattice consistency, constrained cross-layer phase,
  queue-conditioned gating, and reversible local correction. It is not yet a
  novel theory.
- The diagonal-quotient phase signature has an exact known-code isomorphism.
  Map phase vector `phi` to a one-pulse-per-wavelength 2D binary array
  `B_phi(layer, position) = 1[position == phi[layer]]`. A global phase offset
  is a common cyclic column shift, and the PRW overlap `kappa` is exactly the
  shifted cross-correlation of two such arrays. This is the standard orbit and
  correlation structure of OPPW two-dimensional optical orthogonal codes.
- Consequently, standalone phase-address novelty is **refuted**. The dense
  Legendre carrier converts the same OPPW overlap count into a bipolar
  correlation score; it does not create a new address space.
- `typed16` is also known: the eight Walsh/Hadamard rows and their negatives
  are the 16-word first-order Reed-Muller/biorthogonal code `RM(1,3)` with
  parameters `[8,4,4]`.
- The surviving publishable questions are now limited to:
  - a genuinely new finite-sample false-unlock/error theorem for the joint
    dense decoder;
  - a matched-resource Pareto advantage over sparse OPPW and ordinary
    key-value routing;
  - a preregistered positive higher-order interaction that is not explained by
    the known components;
  - a rigorous systems result joining routing, queue control, and reversible
    provenance-scoped correction.
- Hostile implementation review found no P0 defect and independently confirmed
  the FFT shift convention, exact 4691 autocorrelation, quotient gauge,
  overlap bound, `typed16` search, and `free256` factorization. It nevertheless
  issued a NO-GO for PRW-1 through PRW-4 evidence until six P1 contracts close:
  true end-to-end 4096 control, truthful payload storage, crossed planted and
  decoder masks, fixed false-unlock sample/interval rules, distinct clean
  templates and corrupted query types, and a frozen bank argmax/margin/unlock
  evaluator.
- The protocol now freezes a single primary false-unlock cell, disjoint
  hash-derived `BANK`/`SELECT`/`REPORT` streams, exact SELECT/REPORT group
  counts, a fixed score-margin grid, a lexicographic threshold rule, and a
  one-sided 97.5% exact Clopper-Pearson upper bound. Runs below that contract
  are explicitly `SMOKE_NON_EVIDENTIARY`.
- Sparse OPPW decoding over the identical phase codebook is mandatory. Without
  it, any dense-carrier advantage is `INCONCLUSIVE`.
- A deeper reduction further narrows the theorem lane. Flattening every
  type/shift/mask carrier into a bipolar codeword makes dense PRW correlation
  exactly maximum-likelihood nearest-codeword decoding on a binary symmetric
  channel. A fixed competitor at Hamming distance `d` ties or wins exactly
  when at least `ceil(d/2)` of those differing coordinates flip.
- With known matching orientation and `k` aligned layers, the exact Legendre
  distance is `(L-k)*(p+1)/2`. The resulting pairwise error is an ordinary
  binomial tail, and bank error admits the standard distance-spectrum union
  bound. This means a generic finite-noise theorem is coding theory, not a new
  memory theory; only a sharper structured distance spectrum or joint-decoder
  result could remain mathematically interesting.
- An independent small exact check at `p=31`, `L=4`, `k=2`, and bit-flip rate
  `0.35` gave distance `32`, analytic tie-or-win probability
  `0.05784490084795206`, and empirical frequency `0.057965` over 200,000
  trials.
- A post-restart local generic NumPy cyclic-correlation microbenchmark
  (16 candidate types, 8 layers, 30 measured repetitions after warmup) found:
  - length 4091: median `24.863 ms`, p95 `27.657 ms`;
  - length 4096: median `5.351 ms`, p95 `6.859 ms`;
  - length 4691: median `28.347 ms`, p95 `39.702 ms`.
  On this generic CPU path, 4096 is about 5.30x faster than 4691. This is a
  bounded kernel timing, not end-to-end latency and not a Rader implementation,
  but it is direct negative pressure on `PRW-4`.
- The exact code-distance analysis makes the old 90%-at-0.45 target
  scientifically misleading. For `p=4691`, `L=8`, `kappa<=1`, `typed16`, and
  bit-flip rate `0.45`, the wrong-type minimum-distance lower bound is `16418`.
  Across 1,125,840 wrong type/shift/mask hypotheses, the Chernoff union bound is
  about `1.66e-30`. Under the ideal simulator, anything materially below
  perfect raw type recovery is evidence of a bug or violated model assumption,
  not encouraging partial success.
- A sparse OPPW code under the same per-coordinate BSC has distance only
  `2*(8-1)=14` and a large pairwise error at rate `0.45`. That apparent dense
  advantage is the ordinary consequence of expanding the code across roughly
  `p/4` more Hamming distance/energy; it is not a matched-resource memory
  breakthrough.
- The only credible new mathematical conjecture left is sharper and much
  narrower: for the frozen overlap-one Legendre-by-`RM(1,3)` bank, the total
  wrong-type error may asymptotically equal nearest-state multiplicity times
  the exact binomial pairwise tail. Testing it requires disagreement-set
  intersections, Hunter's spanning-tree correction, and importance sampling.
  The pairwise theorem itself is standard; only a proved sharp structured-bank
  asymptotic could be a new decoder result.
- Hostile runner review exposed and killed one tempting but invalid
  composition: applying the type-relative phase signature to the supposedly
  separate payload lets global payload-only search recover the type. Payload
  collision groups must remain in identical type-canonical coordinates and
  inherit only global transport; otherwise there is no carrier "unlock" to
  test.
- A pure shared Legendre ring cannot identify waypoints after maximizing over
  rotation because all rotations are equivalent. It can be a phase/synchrony
  carrier only; identity must live in a separate payload code or subspace.
- Searching all 256 eight-layer polarity masks is algebraically equivalent to
  summing per-layer absolute correlations. It is therefore a false-unlock
  control, not a legitimate default router.
- The existing evidence DAG rejects cycles by design. Any cyclic phase carrier
  must remain inside an experimental router; waypoint lifecycle, correction,
  pruning, and rollback should reuse the existing steering-post bank rather
  than weakening evidence-DAG invariants.
- The proposed prime-phase mechanism therefore has one especially useful
  falsification target: it must make a query-conditioned route decision that
  differs from the static bank while preserving immutable evidence and
  rollback. Merely wrapping the same deterministic posts in rotations cannot
  alter H5 outcomes.
- A later off-baseline Rung-16 campaign is an even closer prior lane:
  `5a3fbcc2` tested a quant-aware subdomain routing plane and failed its frozen
  SELECT gate in both arenas. Plane-minus-single-global SELECT NDCG was
  `0.000000` on SciFact and `-0.000899` on the mixed
  SciFact/NFCorpus/FiQA2018 arena; corresponding REPORT deltas were
  `-0.003332` and `-0.007383`. This is direct negative evidence against the
  assumption that more conditional subdomain routes inherently improve
  retrieval.
- Commit `7e6c9118` records the adjacent living/annealing results:
  - C5 living and C5s static are numerically identical on SciFact
    (`0.131135`) and NFCorpus (`0.046389`);
  - the one-shot C5r route beats living C5 on SciFact and is nearly tied on
    NFCorpus; and
  - the one-seed H4 compound-cycle ablation collapses NDCG from `0.236297` to
    `0.005258`.
  These runs do not test PRW phase coding, but they do falsify any inherited
  premise that layering, living updates, or repeated correction is beneficial.
- Consequently, a PRW key can earn a systems contribution only if its route
  changes real held-out decisions and beats the single-global, direct-metadata,
  static-bank, and one-shot controls. Cleaner indexing of an ineffective bank
  is not a retrieval gain.
- The original `feat/brain-file-map-b0-b1` branch remains preserved. Current
  METHOD_DEV edits are isolated on `codex/prime-ring-onion-method-dev` and are
  intentionally uncommitted while the runner and hostile review converge.
- The strongest newly located primary-source collision is Nguyen Q. A.,
  Györfi, and Massey (IEEE TIT, 1992): it maps each symbol of a `p`-ary outer
  word to a cyclic shift of a length-`p` binary Legendre word. For
  `p = 3 mod 4`, two unequal symbols contribute exactly `(p + 1) / 2` Hamming
  distance, so its unmasked distance is exactly the PRW identity
  `(L - r) * (p + 1) / 2`. This is more direct prior art than the OPPW
  isomorphism alone.
- No primary paper was found in the bounded audit that proves the exact
  `PRW-T1` ratio-one asymptotic for the frozen overlap-one
  Legendre-by-`RM(1,3)` bank. The defensible residual is narrow: freeze `q`,
  tie handling, transmitted-state averaging, and a bank sequence, then prove
  both aggregate nearest-event intersections and all farther-neighbor
  contributions are little-o of
  `A_min * beta_q(d_min)`. Standard pairwise tails, union bounds, and
  conditional "negligible intersections imply exactness" are not novel.
- The bounded exact-number literature audit found no application-specific
  published use of `4691`. That is not evidence that none exists. Its known
  properties remain generic: prime, `3 mod 4`, and a smooth `p - 1` useful only
  if an actual Rader implementation wins a benchmark.
- The final corrected smoke artifact is
  `artifacts/method-dev/prime-ring/prime-ring-waypoint-v1.json`:
  - `576/576` cells retained;
  - `255` completed and `321` structurally unavailable;
  - zero errors and zero false-unlock gate passes;
  - `SMOKE_NON_EVIDENTIARY`, `METHOD_DEV`, promotion and production disabled;
  - `PRW-1` through `PRW-3` inconclusive smoke, `PRW-4` inconclusive,
    `RADER-1` untested, novelty not established.
- The smoke's apparently perfect routed payload result at some cells is not a
  scientific result: there are only three REPORT planted groups per seed,
  thresholds are point-estimate-only, and identical cross-type payloads make
  payload-only Recall@K canonical-tie dependent. The artifact explicitly marks
  `K = 5 > T = 4` as not evidence of type retrieval.
- Required controls remain fail-closed and unimplemented: a native sparse OPPW
  observation with matched noise/energy, an equal-channel-use repeated-bit
  code, block/burst noise, process RSS, repeated hardware timing, and the full
  pooled 512/1024-per-class campaign.
- The actual sample resource accounting now distinguishes, for the
  `p=4691`, `L=8`, `T=4`, `W=4` cell, `4,803,584` logical type-expanded
  payload bytes from `1,200,896` deduplicated stored payload bytes. Actual-cost
  efficiency uses the deduplicated total, while FFT/product/routed temporary
  peaks are conservative componentized estimates rather than process-RSS
  measurements.
- Final live validation passed:
  - `49/49` PRW core, runner, and exact-reduction tests;
  - `31/31` adjacent CRSV adversarial tests;
  - focused Ruff checks clean;
  - stable Clopper-Pearson probes at `n=1536` and `n=3072` invert to CDF
    `0.025` within floating-point tolerance;
  - block-flag gate `CLEAR`, zero carried debt.
- Current verdict: there is no established groundbreaking computing result.
  The broad rotating-prime/onion story reduced to known code constructions and
  ordinary distance expansion. The only mathematically plausible new work is
  the very narrow `PRW-T1` intersection-spectrum theorem; the only plausible
  systems work is a matched-resource carrier-to-separate-payload gate that
  changes real held-out decisions. Neither has been validated.
- Continuation decision on 2026-07-24: resume implementation, exact
  small-prime analysis, and bounded microbenchmarks only. The powered campaign,
  large importance sampling, real-corpus campaigns, and full-size Rader timing
  remain held because they are unnecessary for the next correctness slices and
  could create avoidable memory pressure.
- The next useful work separates into three independent code lanes: matched
  sparse/repetition/noise controls, a resource-guarded `PRW-T1` finite
  intersection enumerator, and a correctness-first bounded Rader
  implementation. Remaining harmonic/polar/3D/queue ideas require typed
  hypotheses before code.
- The remaining informal concepts are now separated in
  `docs/research/prime-ring-remaining-hypotheses-2026-07.md`: graph-coupled
  cyclic fibers (`PRW-G1`), Fourier-phase synchronization (`PRW-H1`), the prime
  dimension null (`PRW-D1`), CRT payload pivots (`PRW-C1`), causal queue
  potential (`PRW-Q1`), and reversible living state (`PRW-L1`). None inherits
  support from the original ring construction.
- RB-2 now has a finite, resource-guarded implementation in
  `prime_ring_intersection.py`. It materializes only bounded small-prime
  Legendre-by-mask banks, computes exact finite BSC pair and pair-intersection
  probabilities, builds the Hunter maximum-spanning-tree correction, and can
  brute-force the true event union only when the full noise-pattern count fits
  the declared budget. It explicitly leaves the asymptotic claim false.
- Eight bounded RB-2 tests pass, including direct noise-pattern oracles for the
  binomial tail, pair-event intersection, and whole-bank union; no importance
  sampling or large-prime bank was run.
- RB-3 has a correctness-first Rader implementation and 13 small-prime tests
  from the independent lane. Only `p=7`, `11`, and `31` correctness paths were
  exercised; no 4091/4691 benchmark was launched.
- The initial RB-1/RB-2 implementations did not survive hostile review
  unchanged. Three RB-1 failures were corrected: fixed-weight noise had been
  mislabeled as an independent BSC, an always-off decoder could close pooled
  controls at zero recall, and resource accounting omitted live temporaries and
  co-resident ring-template arrays while checking too late. RB-2 likewise
  normalized arbitrary inputs before its nominal hard guard.
- RB-1 now uses independent Bernoulli chip flips for both the dense carrier and
  equal-channel-use repeated-bit control. Block and burst paths retain exact
  global counts but are explicitly outside the binomial/BSC calculation. Native
  OPPW remains a distinct symbol-substitution channel, so dense-versus-native
  superiority is not a matched-noise claim.
- Pooled control closure now requires a frozen threshold, the REPORT
  false-unlock confidence gate, and true-unlock recall of at least `0.50`.
  The dense primary path additionally requires 100% raw type accuracy in the
  pooled `q=0.45` cell. The protocol's stronger "through q=0.45" lower-rate
  sweep is not inferred from that endpoint and remains an explicit blocker.
- RB-1 standalone and harness-co-resident NumPy-array estimates are now
  preflighted before native/repeated control allocation under a hard 512 MiB
  ceiling and a 50,000,000-work-unit ceiling. These are conservative array
  estimates, not measured process RSS and not authority for a powered run.
- RB-2 now exposes the full
  `(d_i,d_j,|D_i intersection D_j|)` signature spectrum and nearest-event sum,
  distinguishes competitor tie-or-better unions from final decoder error, and
  accepts PRW-T1-specific labeling only for verified overlap-one,
  `RM(1,3)`, `p congruent to 3 mod 4` banks. Its 512 MiB and 120-second hard
  ceilings cannot be raised by a caller.
- Final bounded validation on the reconditioned tree passed:
  - 61 combined prime-ring core, Rader, and finite-intersection tests;
  - 16 runner tests in 12.299 seconds, using only tiny smoke fixtures and a
    temporary artifact test;
  - 31 adjacent CRSV tests and 4 prior prime-ring theory tests;
  - focused Ruff checks across all six new/changed Python implementation and
    test files.
- RB-3 ultimately has 14 focused tests (not 13) after resource-hostile review,
  plus exhaustive odd-prime property checks through 101 reported by the lane.
  No 4091/4691 transform or benchmark was run.
- The bounded implementation result is still not a breakthrough result:
  PRW-T1 remains a finite diagnostic rather than an asymptotic theorem, Rader
  has no full-size performance evidence, matched controls have no powered
  empirical result, and every novelty/production/promotion gate remains false.

---

## 2026-03-28 Disk-Resident LLM Feasibility Findings
- The paper referenced in the user request was not attached in-session. The closest primary-source fit to the request is `LLM in a Flash`, so the current feasibility pass uses that as the working assumption.
- ChelatedAI's current computational-storage path is still a transport and replay proof, not a transformer runtime.
- `computational_storage_poc/block_graph.py` uses fixed `512 x 512` FP16 dense blocks with zero padding, which is far too inefficient for realistic transformer storage.
- `computational_storage_poc/mock_nvme.py` preloads the full binary into RAM, so the current "NVMe" path does not measure real SSD behavior.
- A credible near-term architecture is SSD-resident compressed weights plus CPU execution, not end-to-end transformer compute inside a commodity SSD controller.
- `LLM in a Flash` is the right architectural template for the storage side:
  - keep attention weights resident
  - stream only a small active FFN slice
  - prefer larger contiguous reads and reuse windows
- `T-MAC` is the right architectural template for the CPU side:
  - execute low-bit kernels directly
  - avoid dequantize-then-matmul overhead
- The new estimator at `computational_storage_poc/disk_llm_estimator.py` shows:
  - dense full-model streaming from SSD is not viable
  - sparse flash-style streaming makes 7B-70B plausible
  - 405B is only marginal even on workstation-class hardware
- Highest-value repo improvements are:
  - manifest-driven quantized packing
  - real disk-backed reads
  - sparse FFN predictor and cache
  - CPU low-bit kernel path
  - transformer microbenchmark acceptance test

## 2026-03-28 Addendum Findings: REAP / TurboQuant / CPU-Disk Systems
- `REAP` is directly relevant for disk-first deployment of coding MoE models because it reduces the stored expert bank without destroying router behavior.
- `TurboQuant` is not a weight-footprint method; it is primarily a KV-cache and vector-search compression method.
- The strongest published CPU-only inference evidence found in primary sources is currently:
  - `T-MAC` for low-bit CPU inference
  - `bitnet.cpp` / `1-bit AI Infra` for native 1-bit CPU inference
  - `Gemma.cpp` as an official CPU runtime surface from Google
- The strongest published evidence for external memory replacing parametric scale is:
  - `kNN-LM`
  - `RETRO`
  - graph-structured retrieval for code generation such as PKG / GraphSkill
- The practical architecture answer is not "SSD instead of GPU" in isolation; it is "small CPU-native core + disk-backed memory and retrieval + optional disk-resident compressed weights."

## 2026-03-28 Revised Roadmap Findings
- The old roadmap framing was too passive: it assumed the main remaining work was evaluation, not a new architecture program.
- The roadmap now needs to be treated as a multi-phase program with explicit dependency ordering:
  - storage substrate
  - CPU inference substrate
  - compression branches
  - retrieval / graph memory
  - runtime integration
  - long-context compression
  - end-to-end promotion review
- ARCH-AEP needs a new mandatory loop for architecture-led work:
  - scope lock
  - implementation
  - ARCH-AEP review
  - code analysis / hardening
  - promote / defer
- The safest way to minimize scope drift is to prohibit cross-phase PRs and require explicit phase non-goals.
- Retrieval / graph memory should be developed as a first-class phase, not mixed into low-level storage or CPU kernel phases.

## 2026-03-29 Phase 1 Storage Substrate Findings
- The first safe implementation slice for `Phase 1` is compatible with the new roadmap:
  - add a manifest-driven packed artifact path
  - switch the mock NVMe path to real file-backed access
  - add a benchmark surface for size/read reduction
- `packed_graph.py` now demonstrates the right substrate direction:
  - explicit manifest
  - exact matrix-shape storage
  - no mandatory `512 x 512` zero padding
  - `mmap`-backed reads
- The legacy padded path remains intact, which keeps the phase bounded and avoids premature runtime integration.
- Moving `MockNVMeDrive` from eager file reads to `mmap` surfaced an important Windows-specific lifecycle concern: tests and call sites must explicitly close the mapping before tempdir cleanup.
- The new storage benchmark showed the intended direction clearly on the current test graph:
  - artifact size reduction: about `94.69%`
  - read-byte reduction: about `94.71%`
  - parity maintained (`max_abs_diff = 0.0`)
- The packed substrate is now wired into the standard compile helpers:
  - `compiler.py --format packed`
  - `train_and_compile.compile_model(..., artifact_format="packed")`
- This phase should stop here for now. The next phase should build on the new packed substrate rather than mixing in CPU kernel work immediately.

## 2026-03-29 Phase 2 CPU Inference Substrate Findings
- The correct next bounded slice after the packed storage substrate is a CPU execution seam that consumes packed artifacts without introducing sparse routing, MoE logic, or retrieval memory yet.
- `cpu_backends.py` now provides a minimal backend abstraction with two baseline implementations:
  - `NumpyFloat32Backend` as the numerical reference path
  - `NumpyInt8DynamicBackend` as a correctness-oriented low-bit baseline
- `packed_cpu_inference.py` proves that packed disk-backed artifacts can be executed through the CPU backend seam without falling back to the legacy padded graph path.
- The current int8 implementation is numerically close enough to serve as a Phase 2 substrate:
  - benchmark `max_abs_diff` is about `0.002349`
  - unit tolerance remained within the current test thresholds
- The current int8 implementation is not yet a performance win:
  - float32 latency was about `0.1624 ms`
  - int8 latency was about `0.3110 ms`
  - reported speedup was about `0.52x`
- That performance result is expected for this implementation shape because it dynamically quantizes both activations and weights on every call and still uses generic NumPy matmul kernels rather than packed low-bit kernels.
- ARCH-AEP promotion decision for this phase:
  - promote as a correctness and interface baseline
  - do not promote as a CPU-performance claim
- The clean dependency-preserving next steps are:
  - `Phase 2b`: packed quantized artifacts plus reusable scales and lower-overhead CPU kernels
  - `Phase 3A`: sparse/dense FFN selective loading on top of the packed storage and CPU seam

## 2026-03-29 Phase 2b Prequantized Artifact Findings
- The cleanest way to improve the initial CPU substrate was to keep the same packed container and add a second packed storage mode rather than inventing a separate artifact family.
- `packed_graph.py` now supports:
  - FP16 packed weights
  - INT8 packed weights with per-block scales
- `packed_cpu_inference.py` now exploits that storage mode by letting the int8 backend consume prequantized weights directly instead of requantizing them on each call.
- `compiler.py` and `train_and_compile.py` now expose `packed_int8` as a first-class artifact format, which keeps the compile surface aligned with the roadmap.
- Validation results for the current microbenchmark are good enough to promote this slice as the first low-bit performance baseline:
  - float32 latency: about `0.1818 ms`
  - dynamic int8 latency: about `0.5022 ms`
  - prequantized int8 latency: about `0.1524 ms`
  - prequantized int8 speedup vs dynamic int8: about `3.30x`
  - prequantized int8 speedup vs float32: about `1.19x`
  - prequantized int8 read bytes: `41600` vs `83200` for the FP16-packed path
  - prequantized int8 max absolute diff: about `0.002579`
- ARCH-AEP promotion decision for this phase:
  - promote the prequantized INT8 path as the current CPU baseline for packed artifacts
  - still treat it as a prototype kernel path, not a final answer for large-model CPU inference
- Remaining technical constraints:
  - activations are still dynamically quantized per call
  - execution still relies on generic NumPy integer matmul rather than packed low-bit kernels
  - no sparse FFN loading, MoE routing, or retrieval memory is included yet
- The next bounded choices remain:
  - `Phase 2c`: deeper CPU-kernel and activation-path optimization
  - `Phase 3A`: selective FFN loading / sparse runtime behavior

## 2026-03-29 Phase 3A Dense / Sparse FFN Findings
- The most bounded way to implement `Phase 3A` in the current POC is to keep the packed artifact format stable and add selective row-chunk reads for streamed blocks.
- `packed_graph.py` now exposes row-chunk readers for both:
  - dequantized float storage
  - prequantized INT8 storage
- `sparse_cpu_inference.py` implements the first selective-loading runtime:
  - resident-vs-streamed split via `stream_from_block`
  - activation-driven routing heuristic using nonzero rows
  - `SparseChunkCache` for chunk reuse across repeated calls
- The current runtime preserves exact linear equivalence by summing only the row chunks whose activations are nonzero, rather than approximating the output.
- The benchmark harness was tuned so the streamed FFN block materially dominates the byte budget; otherwise the always-resident first block hides the gain signal.
- Current Phase 3A benchmark results:
  - avg dense latency: about `0.2478 ms`
  - avg sparse latency: about `0.2253 ms`
  - dense bytes per token: `98304`
  - sparse bytes per token: `37376`
  - streamed byte reduction: about `61.98%`
  - cache hits: `28`
  - cache misses: `36`
  - max absolute diff: about `0.000806`
- ARCH-AEP promotion decision for this phase:
  - promote the selective-loading runtime as the baseline dense/sparse FFN prototype
  - keep the scope explicitly limited to row-chunk streaming on the current toy packed-graph path
- Remaining constraints:
  - routing is still a simple activation sparsity heuristic, not a learned predictor
  - the cache is a small in-memory prototype, not yet a full runtime policy
  - the benchmark is still a layer-execution harness, not transformer token generation
- The clean next bounded choices are now:
  - `Phase 3B`: MoE / REAP-compatible storage and execution
  - `Phase 4`: retrieval and graph memory substrate

## 2026-03-29 Phase 4 Retrieval / Graph Memory Findings
- The cleanest `Phase 4` slice is a standalone repo-memory substrate that does not couple to the inference runtime yet.
- `repo_graph_memory.py` now provides:
  - a disk-backed node / edge / embedding bundle
  - file and symbol nodes
  - local-import and containment edges
  - a memory-mapped embedding surface
  - a query API that blends vector, lexical, and graph signals
- The current embedding approach is intentionally lightweight:
  - stable hashed token embeddings
  - no dependency on external embedding services
  - good enough for local architecture validation
- `repo_graph_memory_benchmark.py` measures this layer independently of the model runtime and currently reports:
  - node count: `111`
  - edge count: `220`
  - ingest latency: about `93.65 ms`
  - average query latency: about `0.4328 ms`
  - top-1 hit rate: `75%`
  - top-3 hit rate: `75%`
- An ingestion filter was necessary to keep the benchmark honest:
  - benchmark files, tests, docs, and cache directories can otherwise dominate repo-local code retrieval with artificial lexical matches
- ARCH-AEP promotion decision for this phase:
  - promote the repo-memory layer as the current disk-backed code-memory substrate
  - keep it decoupled from execution until `Phase 5`
- Remaining constraints:
  - embedding quality is still hash-based rather than model-based
  - graph edges are limited to containment and local-import relationships
  - reranking is heuristic, not learned
- The clean next bounded choices are now:
  - `Phase 5`: integrate storage, CPU, sparse runtime, and repo memory into one runnable prototype path
  - `Phase 3B`: keep MoE / REAP as a separate parallel branch if a concrete MoE target is chosen

## 2026-03-29 Phase 5 Runtime Integration Findings
- The cleanest `Phase 5` slice is a single CPU-only repository Q&A style prototype rather than a fake token generator.
- `integrated_repo_runtime.py` now integrates:
  - the disk-backed repo-memory query surface
  - retrieval-result featureization
  - a packed INT8 reranker artifact
  - the sparse CPU execution path
- The current end-to-end runtime uses a deterministic reranker graph rather than a trained language model, which keeps the phase honest:
  - it proves orchestration and metric capture
  - it does not overclaim generative capability
- `integrated_runtime_benchmark.py` now measures the integrated path and currently reports:
  - average retrieval latency: about `1.3089 ms`
  - average inference latency: about `1.9976 ms`
  - average total latency: about `3.3251 ms`
  - queries per second: about `28`
  - average bytes read per query: `1152`
  - mapped bytes: `121313`
  - peak Python heap: about `55.91 KB`
  - top-1 hit rate: `50%`
  - top-3 hit rate: `75%`
- A code-aware tokenization fix was necessary in the repo-memory substrate so snake_case and CamelCase repository identifiers become searchable in a repo-local way.
- ARCH-AEP promotion decision for this phase:
  - promote the integrated repository-Q&A prototype as the current end-to-end local baseline
  - keep the claim boundary narrow: this is orchestration and retrieval-aware reranking, not a full disk-resident LLM
- Remaining constraints:
  - the reranker is hand-authored, not trained
  - the runtime still targets repository retrieval/reranking rather than free-form code generation
  - no KV-cache or memory compression exists yet
  - MoE / REAP remains intentionally separate
- The clean next bounded choices are now:
  - `Phase 6`: long-context and memory compression on top of the integrated baseline
  - `Phase 3B`: MoE / REAP path as a parallel branch if a concrete MoE target is selected

## 2026-03-29 Phase 6 Memory Compression Findings
- The cleanest `Phase 6` slice on the current baseline is compressed repo-memory embeddings rather than KV-cache work, because the runtime is still a retrieval/reranking prototype rather than a long-context generator.
- `repo_graph_memory.py` now supports:
  - float32 embedding storage
  - int8-compressed embedding storage with a saved scale
- The compression path was threaded through both the standalone repo-memory benchmark and the integrated runtime so memory savings and quality impact can be measured at two levels.
- Current repo-memory compression benchmark results:
  - float32 mapped bytes: `122880`
  - int8 mapped bytes: `30720`
  - mapped-byte reduction: about `75%`
  - float32 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 average query latency is slightly higher in the current path
- Current integrated-runtime compression benchmark results:
  - float32 mapped bytes: `123361`
  - int8 mapped bytes: `31201`
  - mapped-byte reduction: about `74.71%`
  - float32 average total latency: about `3.8535 ms`
  - int8 average total latency: about `4.0467 ms`
  - float32 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 peak Python heap is higher in the current implementation because query-time quantization introduces temporary allocations
- ARCH-AEP promotion decision for this phase:
  - promote the int8 repo-memory path as the current compression experiment baseline
  - do not claim it is fully optimized yet, because the latency and heap tradeoff still need review
- Remaining constraints:
  - compression currently targets the repo-memory embeddings only
  - no KV-cache compression exists because the runtime is not a long-context generator yet
  - query-time quantization overhead still leaves optimization headroom
- The clean next bounded choices are now:
  - `Phase 7`: end-to-end evaluation and promotion review
  - `Phase 3B`: MoE / REAP as a separate parallel branch

## 2026-03-29 Phase 7 Evaluation And Promotion Findings
- `phase7_system_evaluation.py` now aggregates the benchmark pack across storage, CPU, sparse runtime, repo memory, integrated runtime, and compression.
- The current benchmark thresholds all pass for the intended research-baseline scope:
  - storage reduction
  - CPU baseline
  - sparse runtime
  - repo memory
  - integrated runtime
  - standalone compression
  - integrated compression
- The current promotion call is:
  - promote as `research baseline`
  - do not promote as `production-ready`
- The strongest reason for the defer remains architectural scope, not a failing benchmark:
  - the system is still retrieval-and-reranking oriented
  - it is not a full generative runtime
  - it is not yet clearly simpler operationally than a GPU-backed alternative
- The promotion memo at `docs/phase7-promotion-review-2026-03-29.md` now records:
  - promote/defer decision
  - recommended presets
  - no-go conditions
  - next-branch recommendations
- The clean next bounded choices after the closed review are:
  - `Phase 3B`: MoE / REAP branch if a concrete target model exists
  - targeted optimization and evaluation expansion under the current research baseline

## 2026-03-29 Phase 3B MoE / REAP Branch Findings
- The cleanest post-review parallel branch was to keep MoE / REAP isolated from the integrated baseline and land it as a separate artifact/runtime seam.
- `moe_reap.py` now provides:
  - a disk-backed MoE artifact format
  - expert-bank metadata with preserved router ids
  - routed-expert CPU execution
  - a REAP-like pruning compatibility function based on expert weight norms
- The current benchmark proves the intended branch property:
  - full artifact bytes: `2025`
  - pruned artifact bytes: `1122`
  - artifact reduction: about `44.59%`
  - full bytes read: `704`
  - pruned bytes read: `384`
  - read reduction: about `45.45%`
  - full experts evaluated: `4`
  - pruned experts evaluated: `2`
- The branch is intentionally not integrated into the Phase 5 prototype yet.
- ARCH-AEP promotion decision for this branch:
  - promote as a valid parallel MoE / REAP compatibility branch
  - defer runtime integration until a concrete MoE target model is selected

## 2026-03-29 Targeted Optimization And Evaluation Expansion Findings
- The cleanest post-Phase-7 follow-up was a bounded optimization and benchmark-hardening pass on the existing research baseline, not more architecture churn.
- `repo_graph_memory.py` now scores int8-compressed embeddings in row chunks instead of materializing an `int32` copy of the full memory-mapped matrix for every query.
- `repo_graph_memory.py` also now:
  - favors precise path matches over helper-path spillover
  - returns unique paths instead of repeated file/symbol duplicates for the same path
- That change materially improved the compressed integrated-runtime heap profile:
  - previous int8 peak Python heap: about `124.55 KB`
  - current int8 peak Python heap: about `58.45 KB`
- `integrated_repo_runtime.py` now uses retrieval-dominant fusion for final ranking, so the packed reranker refines candidate order instead of overwhelming stronger retrieval evidence.
- `cpu_inference_benchmark.py` now uses warmup plus median-of-trials timing so the Phase 7 promotion call is not driven by one noisy microbenchmark sample.
- `retrieval_eval_suite.py` now defines a shared 10-query repo-local benchmark suite covering:
  - packed storage
  - sparse loading
  - CPU backends
  - training / compile flow
  - repo memory
  - integrated runtime
  - MoE / REAP
  - payload transport
  - emulation
  - disk sizing / feasibility estimation
- The standalone repo-memory benchmark now reports on that wider suite:
  - query count: `10`
  - top-1 hit rate: `100%`
  - top-3 hit rate: `100%`
- The integrated runtime benchmark now reports on the same wider suite:
  - query count: `10`
  - average total latency: about `3.13 ms`
  - top-1 hit rate: `100%`
  - top-3 hit rate: `100%`
- The int8 compression tradeoff is now clearer after the optimization pass:
  - mapped-byte reduction is still about `75%`
  - quality stayed unchanged on the shared suite
  - standalone int8 query latency is now slightly better than float32 on the shared suite
  - integrated int8 total latency remains higher than float32, so compression is still a memory win first and an end-to-end latency win second
- ARCH-AEP decision for this pass:
  - promote as a bounded optimization and evaluation-hardening pass
  - keep the recommendation unchanged: the stack remains a promoted research baseline, not a production-ready system

## Requirements
- Proceed on the active computational-storage follow-up items overnight.
- Use implementation-session style orchestration with research, architecture, implementation, and validation phases.
- Keep track of session log and agent work.
- Create a PR for each reviewable item so the user can review via PR workflow tomorrow.
- Do not jeopardize work running in other repos.

## Research Findings
- Latest active handoff is the computational-storage post-merge follow-up from Session 25.
- The active queue is limited to four items: hardware evidence, emulator-path CI decision, transport-path scope decision, and retention policy review.
- The old Session 21/22 “top 15” is historical and already completed.
- `.github/workflows/test.yml` currently runs global `unittest` plus a dedicated computational-storage fundamentals job, but neither job provides a distinct emulator-path gate.
- `.github/workflows/build_firmware.yml` separately validates RP2040 firmware compilation and artifact generation.
- `test_computational_storage_payload.py` validates the deterministic payload contract, virtual-disk trigger-sector injection, and host-reader decoding from a file path.
- `computational_storage_poc/usb_host_inference.py` is the host-side raw-sector reader for physical or file-backed devices.
- A local hardware probe for RP2040 / Pico / TinyUSB devices returned no present device, so authentic real-hardware evidence cannot be captured tonight unless a device appears later.
- `computational_storage_poc/emulation/fuse_block_emulator.py` already contains emulator semantics, but importing it in CI currently requires `fusepy`; the core read behavior can be extracted into a dependency-light module and tested directly.
- `computational_storage_poc/emulation/docker-compose.yml` requires privileged FUSE (`/dev/fuse`), which is a poor fit for a stable default GitHub Actions gate.
- The existing docs already state that firmware scope is transport correctness, not on-device digits inference, but that boundary can be made more explicit and centralized.
- Backup refs currently present locally include `backup/retired-*` refs plus older local backup refs; remote backup refs from February are also still present.
- The hardware-evidence prep implementation can be validated tonight with file-backed trigger-sector images, which gives strong confidence in tomorrow’s physical capture workflow without claiming hardware success.
- The emulator-CI implementation can stay branch-independent from the hardware-evidence branch by avoiding the optional host-reader verbosity refinement.
- A dedicated emulator job can remain lightweight because it only needs `numpy`, the virtual controller, and the host-reader path.
- The scope decision is best enforced through one canonical doc referenced by the POC and firmware docs, rather than duplicated prose in multiple places.
- The safest retention decision tonight is a timed manual review, not deletion, because the computational-storage follow-up is still active and the March rollback refs are fresh.
- The planning-with-files catchup helper path in the installed skill is stale on this machine (`.claude` path missing), so session recovery has to be done from repo files directly.
- PR `#90` had a real correctness gap during review: `resolve_drive_path()` on Windows rewrote already-formed device paths like `\\.\PhysicalDrive2` into malformed paths. The branch now preserves explicit device paths and has a regression test.
- Local hardware inspection on 2026-03-06 still shows no RP2040 / Raspberry Pi Pico / TinyUSB mass-storage device. The only removable USB disk currently visible is a SanDisk drive, so physical evidence remains blocked.
- PR `#94` should not be merged until after the implementation PRs land or it is refreshed, because its handoff state would otherwise become stale as soon as `#90`-`#93` merge.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Separate active roadmap from historical top-15 work | Prevents wasting time reopening already completed items |
| Use documentation artifacts to preserve “fresh agent” outputs | Maintains auditable orchestration in a single-session environment |
| Treat hardware evidence as blocked by actual device availability | Prevents fabricated or software-only claims being mislabeled as physical validation |
| Slice the work into four item PRs plus one wrap PR | Minimizes conflicts while still giving the user reviewable units tomorrow |
| Prefer a reusable evidence-capture tool over a one-off manual checklist | Lets hardware evidence be captured immediately once a device is available |
| Extend `usb_host_inference.py` with a path resolver and optional verbosity control | Keeps existing CLI behavior while making automation and tests cleaner |
| Keep the emulator-CI branch independent of the host-reader refinement | Avoids hidden coupling between review branches |
| Express the current transport boundary through a canonical decision doc | Makes future scope promotion auditable and reduces overstatement risk |
| Use a tiered retention policy with a dated manual review window | Protects rollback paths now while preventing indefinite artifact sprawl |
| Preserve explicit Windows device paths in `resolve_drive_path()` | Matches the documented CLI contract for hardware evidence capture and avoids mangling valid raw-device arguments |
| Treat current hardware evidence as blocked despite a visible USB disk | The visible removable storage is a SanDisk drive, not an RP2040/Pico-class target |
| Merge the session-wrap PR last | Keeps `next-session.md`, `tracker-pointer.md`, and `CLAUDE.md` aligned with post-merge reality |
| Refresh the existing wrap PR instead of opening a sixth PR | Preserves the existing review thread while replacing the stale pre-merge handoff with the accurate Session 27 state |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No evidence yet that physical RP2040 hardware is available | Verify connected devices before attempting hardware evidence capture |
| `computational_storage_poc/fuse_fs.py` does not exist at the probed path | Inspect actual POC file layout before planning emulator-path CI changes |
| `resolve_drive_path()` mis-handled explicit Windows device paths in PR `#90` | Fix on branch and add regression coverage before merge |
| `ruff check` was mistakenly run against `.github/workflows/test.yml` | Restrict lint to Python files; use other tooling for YAML if needed |

## Resources
- `CLAUDE.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-03-06-session25.md`
- `computational_storage_poc/README.md`
- `computational_storage_poc/firmware/README_FIRMWARE.md`
- `.github/workflows/test.yml`
- `.github/workflows/build_firmware.yml`
- `test_computational_storage_payload.py`
- `computational_storage_poc/usb_host_inference.py`

## Visual/Browser Findings
- None yet.

## 2026-03-06 Roadmap Audit Findings
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md` still shows only `AEP-2026-03-06` as active, and that cycle is limited to real RP2040 hardware evidence capture plus the dated retention review.
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md` aligns with the tracker: no non-hardware engineering phase is listed as active.
- `README.md` says Phases 1-4 are complete.
- `REFACTORING_PLAN.md`, `COMPLETION_SUMMARY.md`, and `PR_DESCRIPTION.md` still contain older "Deferred to Phase 4" or "Future Development" language, so they cannot be treated as authoritative roadmap sources without checking the live code.
- The next audit step is to verify those older deferred items against the current codebase and experiment scripts before declaring the development roadmap complete.
- Live-code verification shows the previously deferred research features are already present:
  - `antigravity_engine.py` implements `ingest_streaming()` and `enable_adaptive_threshold()`.
  - `teacher_distillation.py` supports configurable `batch_size`, chunked encoding, and ensemble parallelism.
  - `cross_lingual_distillation.py` exists with language-aware teacher routing.
  - `online_updater.py` supports `triplet_margin`, `infonce`, and `cosine_similarity` losses plus diagnostics/scheduling.
  - `benchmark_beir.py`, `benchmark_multitask.py`, `dashboard_server.py`, `run_sweep.py`, and `run_large_sweep.py` all exist.
- Historical docs that still list missing features are stale relative to the code. Session 22 and Session 23 logs show the old top-15 implementation items were delivered and merged.
- The notable non-hardware gap is experimental execution, not implementation: `run_large_sweep.py` exists, but `large_sweep_results.json` and `large_sweep_results.csv` do not.
- `docs/phase4-experiment-protocol.md` is useful background for Phase 4 feature usage, but it is not a current post-development roadmap or weight-refinement plan.
- A new canonical current-state doc now exists at `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`.
- `docs/INDEX.md` now links that audit/test-plan doc so future sessions do not need to reconstruct the conclusion from session logs.

## 2026-03-06 Experiment Campaign Findings
- Runtime prerequisites are currently available locally:
  - `torch 2.9.1+cpu`
  - `sentence_transformers 5.2.0`
  - `mteb 2.6.1`
  - `qdrant_client` import succeeded
  - `numpy 2.3.5`
- A `requests` dependency warning is emitted during imports, but it is not a hard blocker.
- Existing local experiment artifacts include `adapter_weights.pt` and `sweep_results.json`.
- No completed large-sweep artifact exists yet.
- `antigravity_engine.py` already exposes `enable_online_updates(...)`, so Phase 5 does not require new engine hooks.
- Historical docs indicate `benchmark_comparative.py` already has an `online_updates` configuration, which may be sufficient for online-ablation work without adding a brand-new benchmark script.
- `benchmark_comparative.py` and `benchmark_beir.py` were not safe to use as-is for real evaluation because their CLIs did not wire in a real `engine_factory`; they defaulted to dummy retrieval. This session patched them to use real engines from the CLI path.
- Real evaluation also required mapping Qdrant point IDs back to original document IDs. A shared `map_predicted_ids()` helper is now in `benchmark_utils.py`.
- `run_sweep.py` and `run_large_sweep.py` originally reused the shared SciFact Qdrant path and had no query cap. This session added:
  - `--max-queries`
  - `--db-path`
- The first bounded campaign failed Phase 1 because an orphaned `run_sweep.py` process from an earlier launch still held `db_scifact_evolution`. That stale process was terminated.
- The campaign runner now:
  - passes `-u` to child Python benchmark commands for unbuffered logging on future runs
  - snapshots/restores `adapter_weights.pt`
  - writes an on-disk manifest
  - uses UTF-8 for child processes
  - launches sweep phases against isolated per-run Qdrant directories
- The current active campaign run is `experiment_runs/weight-refinement-20260306-session28-isolated`.
- During the current isolated run, `phase1_standard_sweep.log` stayed quiet after initialization, but the child `run_sweep.py` process continued consuming CPU and writing to the isolated Qdrant folder, indicating active execution rather than an early crash.
- The worktree currently has uncommitted documentation/planning files from the roadmap audit:
  - `docs/INDEX.md`
  - `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Local branch inventory includes many historical `backup/*`, `feat/*`, `feature/*`, and `pr/*` branches. They will be treated as historical context only unless this campaign needs recovery from them.
- The remaining contamination bug was broader than the runner-level snapshot/restore: `benchmark_distillation.py` reused the shared root adapter across baseline/offline/hybrid in a single process, `benchmark_multitask.py` could reuse it across tasks, and `benchmark_beir.py` inherited the same risk through `ComparativeTestbed`.
- The fix is now centralized in `benchmark_utils.isolated_adapter_state()`. `benchmark_comparative.py` uses it per configuration, `benchmark_distillation.py` uses it per mode, and `benchmark_multitask.py` uses it per task.
- `benchmark_distillation.py` now accepts `--max-eval-queries`, and `run_weight_refinement_campaign.py` passes the campaign query budget through to that phase.
- Targeted validation passed after the fix:
  - `python -m py_compile benchmark_utils.py benchmark_comparative.py benchmark_distillation.py benchmark_multitask.py run_weight_refinement_campaign.py test_benchmark_comparative.py`
  - `python -m ruff check benchmark_utils.py benchmark_comparative.py benchmark_distillation.py benchmark_multitask.py run_weight_refinement_campaign.py test_benchmark_comparative.py`
  - `python -m unittest test_benchmark_comparative.py test_benchmark_beir.py -v`
  - a synthetic real-engine smoke covering baseline/offline/hybrid, with the root adapter checksum unchanged before and after
- The previous run at `experiment_runs/weight-refinement-20260306-session28-isolated` was intentionally abandoned after the contamination diagnosis.
- The fresh clean relaunch is `experiment_runs/weight-refinement-20260306-session28-clean`.
- The clean run started at `2026-03-06T13:42:14` with wrapper PID `50404`, runner PID `86244`, and Phase 1 child PID `72276`.
- Early clean-run evidence:
  - `manifest.json` created successfully with `baseline_adapter_snapshot: null`
  - `phase1_standard_sweep.log` is updating normally
  - Phase 1 is ingesting SciFact into the per-run Qdrant path `experiment_runs/weight-refinement-20260306-session28-clean/qdrant/phase1_scifact_db`
- Completed short-run outputs before the interruption:
  - Phase 1 sweep winner: `learning_rate=0.01`, `threshold=1`, `noise_scale=0.2`, `epochs=5`, improving SciFact NDCG from `0.6289` to `0.6766` (`+0.0477`)
  - Phase 2 distillation produced no retrieval delta across teacher weights `0.3`, `0.5`, and `0.7`; baseline, offline, and hybrid all stayed at mean NDCG `0.7553`, while offline pretraining time ranged from `171.7s` to `216.3s`
  - Phase 3 multitask results were stable (`avg_jaccard=1.0`) but showed zero learning gain; aggregate NDCG was `0.6782` for the small suite and `0.6265` for the medium suite
  - Phase 4 BEIR small favored `baseline` and `random_mask_50pct` (`mean_ndcg_at_10=0.6839`) over `chelation` (`0.5745`) and `online_updates` (`0.5746`), with `online_updates` also incurring materially higher latency (`87.75ms` mean)
- The clean campaign then stranded during `phase4_beir_medium`:
  - existing log content stopped mid-SciFact `online_updates`
  - no `phase4_beir_medium.json`, `phase5_online_ablation.json`, or `SUMMARY.md` was written
  - `manifest.json` stayed frozen at `phase4_beir_small`
- `run_weight_refinement_campaign.py` now supports `--resume-run-dir`:
  - resume mode loads the existing manifest
  - recovers already-completed phases from existing output files
  - executes only missing phases
  - can continue into Phase 5 and launch Phase 6 without replaying the entire campaign
- Resume validation passed with:
  - `python -m py_compile run_weight_refinement_campaign.py test_run_weight_refinement_campaign.py`
  - `python -m ruff check run_weight_refinement_campaign.py test_run_weight_refinement_campaign.py`
  - `python -m unittest test_run_weight_refinement_campaign.py -v`
- Active resumed run state:
  - resume wrapper PID `35364`
  - resumed child PID `47792`
  - child command: `python -u benchmark_beir.py --tier medium --model sentence-transformers/all-MiniLM-L6-v2 --max-queries 50 --output ...\\phase4_beir_medium.json`
  - `logs/phase4_beir_medium.log` was recreated at `2026-03-06 23:39:53`, confirming the resume path re-entered the missing BEIR medium phase

## 2026-03-06 Backlog Triage Findings
- The current repo state still supports the earlier roadmap-audit conclusion: there is no unfinished product-implementation phase outside the computational-storage follow-through.
- `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md` is still aligned with `README.md`: the remaining non-hardware work is experiment execution, result analysis, and optional documentation cleanup.
- The strongest actionable non-test work tonight is cleanup of stale roadmap language in:
  - `REFACTORING_PLAN.md`
  - `COMPLETION_SUMMARY.md`
  - `PR_DESCRIPTION.md`
- Those files still describe deferred "Phase 4" work even though the corresponding capabilities are already present in the codebase, so they now create confusion rather than plan useful implementation work.
- No additional active engineering phase was found via repository-wide backlog-marker search; the remaining concrete product backlog is still:
  - real RP2040 hardware evidence capture when hardware is available
  - the dated retention review window on or after `2026-04-05`

## 2026-03-07 Overnight Orchestration Findings
- The resumed `phase4_beir_medium` process was still live after the priority shifted away from overnight execution. It was intentionally stopped to avoid burning CPU while the session moved to documentation and reviewable PR prep.
- Three isolated PRs were opened instead of mixing code, doc cleanup, and wrap artifacts:
  - PR `#96` `feat: harden weight refinement campaign recovery`
  - PR `#97` `docs: normalize stale roadmap documents`
  - PR `#98` `docs: wrap session28 overnight orchestration`
- PR `#96` contains:
  - real-engine evaluation wiring for comparative and BEIR entrypoints
  - adapter isolation to avoid cross-configuration checkpoint contamination
  - resume support for interrupted campaign runs
  - durable docs for the partial Session 28 campaign results
- PR `#97` converts stale "Phase 4" roadmap language in the old hardening docs into explicit historical notes instead of active backlog wording.
- PR `#98` records the overnight orchestration in the session log, verification log, phase summaries, next-session handoff, and `CLAUDE.md`.
- No test suite was rerun in the overnight session by user direction. Only lightweight non-test validation was recorded:
  - `python -m py_compile ...` for PR `#96`
  - `git diff --check` for PRs `#96`, `#97`, and `#98`
