# Evidence-kernel and masked-subplane METHOD_DEV queue

**Queue ID:** `RB-13`

**Status:** `IMPLEMENTATION_IN_PROGRESS`; the RB-13 dependency queue remains
draft/non-preregistered, the five RB-14 lanes have bounded synthetic harnesses
and sanity artifacts, and RB-15 has frozen only a dependency-light nonlinear-
sidecar Stage-A protocol. No production, corpus, or novelty result is accepted
by this document.

**Date:** 2026-07-27; RB-14 queue extension 2026-08-02; RB-15 optional
subcell extension 2026-08-12

**Scope:** bounded synthetic characterization, falsification, and method
selection for evidence-bearing agent memory, selective routing, graph
propagation masks, query-reset local attenuation sidecars, heterogeneous local
representation spaces, and bounded off-span/coalitional observability

**Scientific claim status:** unconfirmed

**Novelty claim status:** unconfirmed. Bitemporal memory, provenance graphs,
learning to defer, graph filters, cellular sheaves, lattice-valued state,
submodular culling, and Grassmann optimization are established areas. Only a
specific integration with a measured advantage over information-, parameter-,
storage-, and operation-matched controls could remain a candidate contribution.

## 1. Scope lock and interpretation

This queue translates the current language into independently testable
operators. It does not treat metaphors as mechanisms:

| Phrase | Testable interpretation |
|---|---|
| deference | select which decision route answers, retrieves, verifies, invokes a tool, escalates, or abstains |
| deflection | attenuate or redirect propagation from dependent, contradicted, uncertified, or unsafe sources |
| reverberation | repeated application of a declared graph operator |
| culling mask | an ephemeral query-time or optimization-time node/feature selection operator |
| subplane | a declared local vector space, subspace, or coordinate chart |
| unified area | a compatible global section or fiber-product state across local spaces |
| resonance | prohibited unless a declared mode/frequency predicts held-out amplification; otherwise use `repeated propagation` or `transient amplification` |
| nonlinear neutraliser | a query-reset auxiliary state with declared attachment map, energy/boundedness screen, and branch policy |

For this queue, “Danto” is interpreted as the public **Donto** evidence-memory
line because that is the agent/brain architecture discussed immediately before
this queue. If a different Danto was intended, it is not silently substituted
into these hypotheses.

This work must not reopen prior closed claims:

- `PRW-JO1-STACKING-MECHANISM` remains
  `CLOSED_BY_FLAT_IDENTITY`. A fixed stack with the same observations and
  decoder opportunity is one flat map/code.
- Existing CRSV/SRS-1 work already implements principal-angle diagnostics,
  signed additive interference, noncommuting order effects, finite-horizon
  amplification, ordered products, and useful-signal atrophy on constructed
  examples. RB-13 may reuse those diagnostics, but it may not rename them as
  new evidence.
- Repeated traversal is not repeated observation. A cyclic path can improve
  retrieval propagation but cannot create independent corroboration.
- Low cross-layer disagreement is not truth. Multiple wrong layers can agree.
- Query-time masks cannot delete, overwrite, or silently demote the stored
  evidence ledger.
- `PRW-G2` already establishes the narrow existence of pair and strict
  three-way interaction residuals on its frozen algebraic fixture while
  recording no static advantage. RB-14 must not rename or rerun that result as
  new "synergy" evidence; `PRW-COA1` tests the distinct missing question of
  intervention-design observability.
- The RB-14 additions are a crosswalk from PCHO-adjacent language, not evidence
  that PCHO is implemented here. `PRW-OBS1` concerns prospective support
  expansion, while `PRW-COA1` and `PRW-CTX1` concern conditional joint tests.
  The original dormant-branch re-observation hypothesis retains its own
  disposition.
- RB-15 transfers a nonlinear-neutraliser mechanism into a query-reset
  auxiliary graph state. It does not establish off-span observability, unknown
  direction discovery, objective truth, or evidence independence. Its full
  source and prior-art audit is
  `docs/research/nonlinear-neutraliser-subspace-transfer-2026-08.md`.

RB-13 is an umbrella queue, not one coupled experiment. `PRW-EK*` cards concern
the certified evidence-kernel lane. `PRW-BIL1`, `PRW-DDF1`, `PRW-RCM1`,
`PRW-RCM1-NLN`, `PRW-ISI1`, `PRW-SPU*`, `PRW-VAR1`, and `PRW-REV1`
concern cross-cutting or masked-subplane operators. A result in one family
supplies no evidence for another. Before execution, each card or tightly
coupled family requires its own frozen protocol, inputs, SELECT/REPORT split,
and disposition.

The existing CRSV predecessor boundary is stricter than a citation:

> CRSV-1, LIR-1, and SRS-1 are frozen predecessor baselines. Principal angles,
> signed additive interference, commutators, finite-horizon gain,
> ordered-prefix products, useful-signal atrophy, and the existing scale sweep
> are not RB-13 contributions and may not be renamed as deference, deflection,
> reverberated culling, subplane dynamics, or variational behavior. An RB-13
> cell is admissible only if it adds a declared state-, query-, or
> evidence-dependent selection operator, a fixed information and search
> budget, an intervention response, and held-out incremental predictive or
> utility value beyond SRS-1. A constant mask, fixed projector, diagonal
> rescaling, last-write-wins overwrite, or flat concatenation is an existing
> reduction, not a new mechanism.

### 1.1 Card status and dependency source of truth

`QUEUED_METHOD_DEV` means the next step is to freeze a runnable protocol; it
does not mean implemented, preregistered, or executed. The table below is the
RB-13/RB-14 dependency source of truth:

| Card | Status | Required predecessor disposition(s) |
|---|---|---|
| `PRW-EK0` | `QUEUED_METHOD_DEV` | none |
| `PRW-BIL1` | `QUEUED_METHOD_DEV` | none |
| `PRW-SPU0` | `QUEUED_METHOD_DEV` | none |
| `PRW-VAR1` | `QUEUED_METHOD_DEV` | none |
| `PRW-OBS1` | `QUEUED_METHOD_DEV` | none |
| `PRW-COA1` | `QUEUED_METHOD_DEV` | none |
| `PRW-EK1` | `BLOCKED_ON_PRW-EK0` | `PRW-EK0` |
| `PRW-EK2` | `BLOCKED_ON_PRW-EK1` | `PRW-EK1` |
| `PRW-EK3` | `BLOCKED_ON_PRW-EK1` | `PRW-EK1` |
| `PRW-EK4` | `BLOCKED_ON_PRW-EK3_AND_PRW-BIL1` | `PRW-EK3`, `PRW-BIL1` |
| `PRW-EK5` | `BLOCKED_ON_PRW-EK1_PRW-EK2_AND_PRW-BIL1` | `PRW-EK1`, `PRW-EK2`, `PRW-BIL1` |
| `PRW-EK6` | `BLOCKED_ON_PRW-EK3` | `PRW-EK3` |
| `PRW-DDF1` | `BLOCKED_ON_PRW-EK2` | `PRW-EK2` |
| `PRW-RCM1` | `BLOCKED_ON_PRW-EK2_AND_PRW-BIL1` | `PRW-EK2`, `PRW-BIL1` |
| `PRW-RCM1-NLN` | `BLOCKED_ON_PRW-RCM1_AND_PRW-ISI1` | `PRW-RCM1`, `PRW-ISI1`; frozen Stage-A mathematical sanity may run independently |
| `PRW-ISI1` | `BLOCKED_ON_PRW-EK3` | `PRW-EK3` |
| `PRW-SPU1` | `BLOCKED_ON_PRW-SPU0` | `PRW-SPU0` |
| `PRW-CTX1` | `BLOCKED_ON_PRW-COA1_AND_PRW-EK2` | `PRW-COA1`, `PRW-EK2` |
| `PRW-SPU1-TRANSPORT` | `BLOCKED_ON_PRW-SPU1` | `PRW-SPU1`; confound-control subcell only |
| `PRW-REV1` | `BLOCKED_ON_PRW-EK1_AND_PRW-RCM1` | `PRW-EK1`, `PRW-RCM1` |
| `PRW-EK7` | `BLOCKED_ON_REQUIRED_COMPONENT_DISPOSITIONS` | unchanged original RB-13 component dispositions plus the RSS/checkpoint guard; RB-14 optional subcells are excluded unless carried into the pilot |
| `PRW-EK7-COALITION` | `BLOCKED_ON_PRW-COA1_SANITY_PRW-CTX1_DISPOSITION_AND_PRW-EK7_ENTRY_GATE` | `PRW-COA1` required sanity boundary, explicit `PRW-CTX1` disposition, and the unchanged `PRW-EK7` entry gate; optional survivor ablation |

The table's `Status` column is the dependency/entry status, not a claim that
the card has no code. The RB-14 implementation overlay is recorded in each
card below: OBS1, COA1, CTX1, the transport subcell, and the coalition
surrogate are implemented with exact bounded sanity artifacts, while their
full predecessor and candidate-survival gates remain open.

`PRW-RCM1-NLN` is an RB-15 optional subcell. Its source audit and exact
Stage-A fixture are frozen, but the parent dependency status remains blocked.
Stage A can validate only the numerical implementation, linear limit, passivity
screen, hardening-response fixture, protected-channel decoupling, and matched
attachment comparisons. It cannot validate graph-RAG utility or novelty.

Every RB-14 addition has this status contract unless its card states the more
specific implementation boundary:

- **Relationship to existing work:** `NEW_CONTRAST_REUSES_PREDECESSOR`.
- **Prior evidence disposition:** `PRESERVE_UNCHANGED`.
- **Evidence state:** `PROPOSED` until its bounded harness has run; the
  2026-08-04 artifacts below use `VALIDATED` only for the exact synthetic
  sanity boundary.
- **Protocol status:** `NOT_FROZEN` unless the card-specific bounded sanity
  harness is named below.
- **Execution status:** `NOT_RUN` for candidate-survival work; the card
  overlays below record the bounded synthetic sanity execution separately.
- **Scientific claim status:** `UNCONFIRMED`.
- **Novelty claim status:** `UNCONFIRMED`.

The 2026-08-04 bounded pass implemented and ran all five RB-14 harness slices
without changing the production retrieval path. The durable artifact manifest
is `artifacts/method-dev/rb14-observability/manifest.json`; its five stage
artifacts all report `scientific_claim_status=UNCONFIRMED`. Candidate-survival
contrasts, independent confirmation sets, predecessor dispositions, and the
unchanged `PRW-EK7` entry gate remain outstanding.

### 1.2 Current repository boundary

The current implementation is a useful baseline, but it is not already the
proposed evidence kernel:

- `ModelScopeMemoryStore._append_entry` removes the oldest entry at capacity;
- `annotate_entry` updates a stored payload in place;
- `record_observation(promote=True)` can write persistent memory without
  evaluating `promotion_status`;
- `promote_episode` accepts an absent promotion status;
- `PersistentMemory` overwrites and deletes records by key;
- `evidence_contract.py` records one `created_at` timestamp rather than
  independent valid and transaction time;
- the current evidence DAG is a structural query/cluster/actuator DAG, not a
  claim/evidence/certificate/authorization graph; and
- the promotion contract does not impose the source-independence, freshness,
  invariance, intervention-response, circularity, and authority gates proposed
  here.

These behaviors are not declared bugs against their existing cache and
promotion contracts. They are compatibility gaps relative to RB-13.
`PRW-EK0` must characterize them before any replacement, adapter, or migration
is designed.

## 2. Formal object

### 2.1 Evidence-bearing claim

A claim record is

\[
c=(\phi,\tau,\Gamma,T_v,T_x,\Pi,W,\mathcal G,\mathcal I,q,p,a),
\]

where:

- \(\phi\) is machine-readable claim content and \(\tau\) is its claim type;
- \(\Gamma\) records explicit assumptions and scope;
- \(T_v\) and \(T_x\) are valid-time and transaction-time intervals;
- \(\Pi\) is the provenance/dependency lineage;
- \(W\) is the evidence and certificate set;
- \(\mathcal G\) is a typed set of nuisance-preserving transformations;
- \(\mathcal I\) is a typed set of material interventions;
- \(q\) is an epistemic state that preserves unknown, supported, refuted, and
  conflicted as distinguishable states;
- \(p\) is maturity/promotion state; and
- \(a\) is action authority, kept separate from factual support.

The proposed four-state evidence bilattice encodes

\[
q=(s,r)\in\{0,1\}^2
\]

as unknown \(N=(0,0)\), supported-only \(T=(1,0)\), refuted-only
\(F=(0,1)\), and conflicted \(B=(1,1)\). Its knowledge order is componentwise:

\[
(s,r)\leq_k(s',r')\iff s\leq s'\land r\leq r',
\]

with \(\wedge_k=\operatorname{componentwise\ min}\) and
\(\vee_k=\operatorname{componentwise\ max}\). Its truth order is:

\[
(s,r)\leq_t(s',r')\iff s\leq s'\land r'\leq r,
\]

with

\[
(s,r)\wedge_t(s',r')=(\min(s,s'),\max(r,r'))
\]

and

\[
(s,r)\vee_t(s',r')=(\max(s,s'),\min(r,r')).
\]

This state records available support/refutation, not objective truth. Merge
order must not change the result for the same active event set. Retraction and
correction append temporal events and derive a new active set rather than
rewriting a prior lattice state. `PRW-BIL1` tests these semantics instead of
assuming them.

No weighted score may substitute for a failed mandatory gate. A candidate
admission predicate is:

\[
\operatorname{Admit}(c) \iff
V_\tau(c,W,\Gamma)\land
\operatorname{Independent}(\Pi)\land
\operatorname{Invariant}_{\mathcal G}(c)\land
\operatorname{Sensitive}_{\mathcal I}(c)\land
\neg\operatorname{Circular}(\Pi).
\]

This predicate is a hypothesis to test, not a truth oracle. In particular,
invariance under paraphrase or coordinate change is not factual truth, and
intervention sensitivity is not proof of causality.

### 2.2 Deference and deflection

Deference is a selective decision policy:

\[
\pi(q,s)\in
\{\text{answer},\text{retrieve},\text{verify},\text{tool},
\text{human},\bot\},
\]

\[
\pi^*=\arg\min_\pi
\mathbb E[\ell(y,\hat y_{\pi(q,s)})+c(\pi(q,s))]
\quad\text{subject to declared risk/coverage constraints}.
\]

Deflection is a propagation operator:

\[
B_{ij}=\widetilde A_{ij}\kappa_{ij},
\qquad 0\leq\kappa_{ij}\leq1,
\]

where \(\kappa_{ij}\) attenuates copied, uncertified, contradicted, or
common-cause-dependent factual messages. Deference decides who investigates or
answers; deflection changes factual-message propagation. Action authority is a
separate `PRW-EK4` gate and must never attenuate factual support. One learned
gate must not both manufacture its own evidence and validate the resulting
decision.

### 2.3 Reverberated culling masks

Two channels must be separated. For seed signal \(z\), ordinary bounded linear
propagation computes query relevance only:

\[
u_K=\sum_{k=0}^{K}\theta_k B^k z.
\]

It is not an evidence count. Independent provenance is carried separately as a
set of primary/common-cause lineage identifiers:

\[
\Lambda_{k+1}(i)=
\Lambda_k(i)\cup
\bigcup_{j:B_{ij}\neq0}\Lambda_k(j).
\]

Set union is idempotent, so a lineage returning through a cycle remains one
lineage. Support/refutation is aggregated at most once per lineage into the
four-state evidence view \(q_i=(s_i,r_i)\). A pre-collapsed source-genealogy
graph is a required control. No scalar \(B^k z\) value may be promoted into an
independent-witness count.

The relevance-derived mask is:

\[
m_i=\mathbf 1[u_{K,i}\geq\tau],
\qquad
M=\operatorname{diag}(m).
\]

The following architectures are not equivalent and must be compared:

\[
M g(B)z,\qquad
g(MBM)Mz,\qquad
u_{k+1}=(1-\alpha)z+\alpha B M_k u_k.
\]

For the fixed-mask geometric recurrence
\(u_{k+1}=(1-\alpha)z+\alpha BMu_k\), Neumann convergence requires
\(\rho(\alpha BM)<1\). That condition does not cover an arbitrary infinite
series or an adaptive mask. The lineage-set recurrence is monotone over a
finite lineage universe and terminates at a set fixed point; an adaptive
relevance mask is nonlinear and requires explicit convergence, two-cycle, and
longer-cycle detection. A monotone culling mask terminates but may be
path-dependent and irreversibly hide weak bridge facts.

Support and refutation remain two channels, so a contested claim does not
collapse into the same state as an unknown claim. A node is eligible for
culling only when query relevance is below threshold and both evidence channels
are weak under the declared query, never because \(s_i-r_i\) is near zero.

### 2.4 Heterogeneous subplanes and unification

Let an evidence graph be \(G=(V,E)\). Assign each node a local vector space
\(\mathcal F(v)\), each overlap an edge space \(\mathcal F(e)\), and
compatibility maps \(\rho_{v\to e}\). A unified state is a global section:

\[
\Gamma(\mathcal F)=
\left\{
x:
\rho_{u\to e}x_u=\rho_{v\to e}x_v
\quad\forall e=(u,v)
\right\}.
\]

Approximate incompatibility is measured by:

\[
E_{\mathcal F}(x)=
\frac12\sum_{e=(u,v)}
w_e\left\|
\rho_{u\to e}x_u-\rho_{v\to e}x_v
\right\|^2
=\frac12x^\top L_{\mathcal F}x.
\]

For a small number of layers, a fiber product is the simpler control:

\[
X_1\times_B\cdots\times_BX_L
=
\{(x_1,\ldots,x_L):q_1(x_1)=\cdots=q_L(x_L)\}.
\]

Fixed linear projections do not establish a stacking mechanism:

\[
P_KP_{K-1}\cdots P_1x=P_{\mathrm{flat}}x.
\]

The product is one linear map whether or not it is itself an orthogonal
projector. A new subplane lane can survive only through a declared difference
such as heterogeneous local dimensions, missing views, data-dependent routing,
nonlinear state, time dependence, versioned coordinate maps, or a proved
representation/decoder cost advantage. It must still beat the corresponding
flat and established multi-view controls.

### 2.5 Intervention support and coalitional observability

Let (c_t) be context, (z_t\in\{0,1\}^d) an intervention or active-memory
mask, and \(\phi_q(c_t,z_t)\) a declared feature lift containing context-weighted
main effects and interactions through order (q). The discounted design is:

\[
V_T^\lambda=
\sum_{t=1}^{T}\lambda^{T-t}
\phi_q(c_t,z_t)\phi_q(c_t,z_t)^\top.
\]

Only directions in \(\operatorname{range}(V_T^\lambda)\) are identified by the
logged interventions. A direction in \(\ker(V_T^\lambda)\) is not negative; it
is unidentified. A target-space coverage condition is:

\[
\forall h\ne0,\qquad
\sum_t
\mathbb E\!\left[
(h^\top\phi_q(c_t,z_t))^2
\mid\mathcal F_{t-1}
\right]=\infty.
\]

Finite-time estimation additionally requires a declared restricted-eigenvalue
or minimum-eigenvalue bound. Boundedness, randomness, or oscillation alone does
not imply or refute this condition.

For strictly positive weights \(w_t\),

\[
\operatorname{range}\!\left(\sum_t w_t x_t x_t^\top\right)
=\operatorname{span}\{x_t\}.
\]

Scalar decay therefore changes conditioning but creates no new direction. If
all candidate generation, momentum, and perturbations preserve a strict
subspace \(S\), two response functions can agree on every observation in \(S\)
and disagree outside it. `PRW-OBS1` must include a bounded full-rank control so
that this invariant-span impossibility is not misstated as a bounded-space
impossibility.

If every mask satisfies \(\lVert z_t\rVert_0<q\), every pure order-\(q\)
interaction feature is identically zero. In particular, singleton probing can
span the raw coordinate space while remaining exactly blind in the lifted pair
space. `PRW-COA1` tests this observation-design ceiling; it does not rediscover
the already implemented Möbius interaction coefficient.

Context-specific positives also need not compose:

\[
\forall i\;\exists c_i:\Delta_i(c_i)>0
\quad\not\Rightarrow\quad
\exists c:\Delta_{\{1,\ldots,k\}}(c)>0.
\]

`PRW-CTX1` therefore requires joint tests under one compatible logged context
and propensity support. It may not add favorable singleton effects obtained
under different contexts or model versions.

Finally, before temporal decay or novelty scoring compares embeddings from
different checkpoints, the protocol must distinguish a basis rotation,
permutation, or bridge change from semantic drift. Alignment can remove a
coordinate confound; it cannot establish factual truth or off-support value.
`PRW-SPU1-TRANSPORT` is limited to that confound-control question.

## 3. Variational-method boundary

Gelfand and Fomin's calculus of variations and the classical Euler-Lagrange
equation are not incorrect. For a smooth function or field with functional

\[
\mathcal J[f]=\int_\Omega L(x,f,\nabla f)\,dx,
\]

the corresponding function-space Euler-Lagrange equation is appropriate.
That machinery is incomplete or category-mismatched for a system containing
binary masks, top-\(k\) selection, graph rewiring, variable dimensions,
provenance lineages, and lattice-valued evidence status.

For a finite graph/sheaf state,

\[
E(x)=\sum_i\phi_i(x_i)+\frac{\lambda}{2}x^\top L_{\mathcal F}x
\]

has the ordinary finite-dimensional first-order stationarity condition:

\[
\nabla\phi(x)+\lambda L_{\mathcal F}x=0.
\]

This is a discrete variational/gradient equation, not the classical continuum
Euler-Lagrange equation. A stationary point is not necessarily a minimum and
is not a truth certificate. The full problem is mixed:

\[
\min_{m,x,\{U_v\}}
\mathcal L_{\mathrm{task}}
+\lambda E_{\mathcal F}(x,m)
+\gamma\operatorname{TV}_G(m)
+\eta C_{\mathrm{dependency}}(m)
\]

subject to:

\[
m\in\{0,1\}^{|V|},\qquad
\|m\|_0\leq B,\qquad
U_v\in\operatorname{Gr}(r_v,d_v).
\]

The method must follow the variable type:

| Component | Candidate method |
|---|---|
| smooth function/field over a continuum | Euler-Lagrange or function-space gradient equations |
| finite graph/sheaf state | finite-dimensional gradient and graph/sheaf Laplacian methods |
| \(L_1\) or graph total variation | subgradient, proximal, or primal-dual methods |
| binary/top-\(k\) mask | graph cut, submodular optimization, or MILP |
| plane orientation | Grassmann/Stiefel Riemannian optimization |
| partially ordered evidence state | lattice theory and fixed-point methods |
| sequential query/tool choice | dynamic programming or optimal control |

Euler-Lagrange is therefore available only for a declared continuum relaxation,
not as the default optimizer for the whole architecture. `PRW-VAR1` measures
the relaxation and rounding gap instead of assuming it away.

## 4. Experiment queue

No card below has been executed.

### `PRW-EK0` — current memory-contract characterization

**Purpose:** establish the code-backed baseline before designing a Donto-like
or evidence-kernel replacement.

**Current implementation questions:**

- Does bounded `ModelScopeMemoryStore` FIFO-delete overflow?
- Does annotation mutate a stored payload rather than append a new version?
- Can `record_observation(promote=True)` persist without evaluating a
  promotion gate?
- Does a missing promotion status pass `promote_episode`?
- Does `PersistentMemory` overwrite and delete values by key?

**Method:** add isolated characterization tests using temporary stores only.
Exercise overflow, annotation, direct promotion, missing promotion status,
repeated saves, deletion, restart, and bounded interleavings. Freeze observed
behavior as `CURRENT_CONTRACT`, `BUG`, or `MIGRATION_GAP`; do not change
behavior in this card.

**Null:** the current stores already provide append-only bitemporal,
provenance-preserving, gate-enforced semantics.

**Kill/route condition:** if the null survives direct code tests, remove the
claimed prerequisite gap. Otherwise require an explicit compatibility and
migration design before any production replacement.

**Ceiling:** one process, under 256 MiB RSS, under two minutes.

### `PRW-BIL1` — epistemic bilattice merge and fixed-point audit

**Purpose:** test whether the declared four-state evidence bilattice preserves
unknown/conflicted distinctions and deterministic merges better than scalar or
order-dependent state.

**Dependencies:** none.

**Fixture:** versioned support and refutation events with duplicates,
independent and copied lineages, concurrent arrival orders, cycles,
retractions, corrections, and valid-time changes.

**Candidate:** the Section 2.1 knowledge/truth orders with idempotent lineage
aggregation and temporal active-set derivation.

**Controls:** scalar net support \(s-r\), last-write-wins categorical state,
paired support/refutation booleans with order-dependent overwrite, and simple
paired-boolean set union.

**Primary metrics:** unknown/conflict confusion, permutation invariance,
idempotence, fixed-point convergence, current/as-of reconstruction, false
promotion, and operations.

**Null:** the declared bilattice provides no semantic or decision advantage
over the strongest paired-boolean control.

**Pass boundary:** every permutation of the same active event set yields the
same state; unknown and conflicted are never confused; duplicate/cyclic
lineages do not change support; finite monotone propagation reaches the same
fixed point; and temporal views match the generator oracle.

**Kill/route condition:** any order-dependent merge, unknown/conflict collapse,
or nonterminating finite monotone propagation rejects the proposed semantics.
If simple paired-boolean set union matches every result and cost, retain that
minimal known representation and drop any separate bilattice mechanism claim.

**Ceiling:** at most 4,096 events and 100 small cyclic graphs, under 256 MiB
RSS, under two minutes per cell.

### `PRW-EK1` — append-only bitemporal correction ledger

**Purpose:** test the compatibility capability of immutable event history plus
query-time current/as-of views. This card does not test typed gates or claim a
research benefit.

**Dependencies:** `PRW-EK0`.

**Fixture:** synthetic claims with delayed observations, corrections,
retractions, transaction-time delay, and valid-time disagreement.

**Method and controls:** hold claim typing and gate behavior fixed. Shadow each
operation into the current store, a last-write-wins key/value view, and an
append-only untyped event ledger. Compare supported present-state API views
with the current store; compare current/as-of ledger views with the fixture's
temporal oracle. Inject failure between event append and view refresh and replay
from the ledger.

**Primary metrics:** present-view parity, current/as-of reconstruction,
retraction handling, raw-evidence hash preservation, deterministic replay,
bytes, update latency, and query latency.

**Null:** an append-only untyped bitemporal ledger cannot reproduce all
supported present-state views and exact temporal-oracle views within the
declared compatibility cost.

**Pass boundary:** 100% parity for supported present-state views, 100% expected
current/as-of reconstruction, zero changed raw-evidence hashes, deterministic
crash replay, at most 1.5x p95 latency, and at most 3x storage in the bounded
prototype.

**Kill condition:** any supported present-state or temporal-oracle mismatch,
altered raw-evidence hash, ambiguous replay state, hidden metadata advantage, or
resource-bound violation.

**Ceiling:** at most 5,000 claims and 25,000 events, under 512 MiB RSS, under
five minutes per seed.

### `PRW-EK2` — copied-source and common-cause collapse

**Purpose:** prevent duplicated reports from becoming false corroboration.

**Dependencies:** `PRW-EK1`.

**Fixture:** provenance DAGs with independent sources, exact copies,
paraphrased copies, hidden common causes, cycles, and one high-volume false
cluster.

**Controls:** majority vote, scalar confidence, source-count deduplication,
explicit lineage collapse, and an oracle dependency partition reported only as
an upper bound.

**Primary metrics:** false admission, copied-evidence amplification,
independent-support recall, calibration, and sensitivity to missed edges.

**Null:** lineage-aware collapse does not beat calibrated confidence plus exact
duplicate removal.

**Pass boundary:** at least 90% reduction in descendant-echo false
corroboration, at least 98% retention of independent support, at most 2%
over-culling, and bounded near-linear traversal on the frozen generator.

**Kill condition:** repeated walks or paraphrases increase the independent
witness count, or the gain exists only with oracle lineage.

**Ceiling:** at most 5,000 nodes and 50,000 edges, under 768 MiB RSS, under
five minutes per seed.

### `PRW-EK3` — typed mandatory gates versus one weighted score

**Purpose:** determine whether non-substitutable claim-type gates prevent
unsafe promotion without unacceptable coverage loss.

**Dependencies:** `PRW-EK1`.

**Fixture:** formal, computational, empirical, normative, and action-authority
claims with controlled missing, circular, stale, or contradictory evidence.

**Controls:** one calibrated weighted score, a rule list without claim types,
typed mandatory gates, and typed gates with abstention.

**Primary metrics:** false promotion, false rejection, abstention, coverage,
calibration, and cost.

**Null:** typed gates provide no risk/coverage advantage over a calibrated
single score.

**Pass boundary:** at least 25% relative false-promotion reduction against the
strongest baseline at matched clean recall, no more than two percentage points
of clean-recall loss, zero invalid formal proofs promoted, and zero stale state
claims presented as current on the frozen safety cases.

**Kill condition:** the reported gain is only a stricter threshold at lower
coverage or uses the target label in gate construction.

**Ceiling:** at most 10,000 claim decisions, under 512 MiB RSS, under five
minutes per seed.

### `PRW-EK4` — factual support versus action authority

**Purpose:** test the required separation between “probably true” and
“authorized to act.”

**Dependencies:** `PRW-EK3` and `PRW-BIL1`.

**Fixture:** true-but-unauthorized memories, authorized-but-stale instructions,
source laundering, revocations, and conflicting principals.

**Controls:** truth confidence alone, one joint score, two-channel
truth/authority gates, and an oracle policy reported only as an upper bound.

**Primary metrics:** unsafe action rate, correct-action coverage, abstention,
revocation latency, and utility.

**Null:** separating factual support and authority does not reduce unsafe
actions at matched useful coverage.

**Pass boundary:** zero unauthorized or stale high-risk executions, 100%
version-substitution/TOCTOU detection, and no more than 5% additional
unnecessary deferral versus the strongest safe baseline on the frozen cases.

**Kill condition:** the split leaks authority labels into truth evaluation or
reduces unsafe actions only by refusing nearly everything.

**Ceiling:** at most 10,000 decisions, under 512 MiB RSS, under five minutes per
seed.

### `PRW-EK5` — local demotion and dependency blast radius

**Purpose:** test whether a counterexample can demote only dependent claims
while preserving unrelated history and support.

**Dependencies:** `PRW-EK1`, `PRW-EK2`, and `PRW-BIL1`.

**Fixture:** versioned claim/evidence DAGs with corrections, retractions,
shared premises, and deliberately missing dependency edges.

**Controls:** full rebuild, recursive invalidation, bounded lineage invalidation,
and no demotion.

**Primary metrics:** correct demotions, collateral demotions, stale survivors,
repair latency, invalidated nodes, and historical query accuracy.

**Null:** local lineage-aware demotion offers no accuracy/cost advantage over a
full rebuild.

**Pass boundary:** exact final-state parity with full recomputation, zero
unsupported survivors, zero non-descendant demotions, and at most 25% of
full-recompute node/edge work under sparse invalidation.

**Kill condition:** missed lineage leaves unsafe survivors or cycles make the
result order-dependent without a declared fixed-point policy.

**Ceiling:** at most 10,000 nodes and 50,000 edges, under 768 MiB RSS, under
five minutes per seed.

### `PRW-EK6` — reversible language projection

**Purpose:** test language as a reversible query/rendering view over canonical
claims rather than as the authoritative store.

**Dependencies:** `PRW-EK3`.

**Fixture:** canonical typed claims, paraphrases, translations, lossy
summaries, scope changes, negations, and value/unit changes.

**Controls:** raw-text memory, canonical record only, render/parse round trip,
and independent semantic/metamorphic checks.

**Primary metrics:** exact-field round-trip rate, semantic false pass, semantic
false fail, scope loss, retrieval accuracy, and storage.

**Null:** reversible projection does not improve update consistency or
retrieval over raw text plus embeddings.

**Pass boundary:** 100% required-field preservation for declared lossless
transformations, 100% rejection of protocol-declared safety-critical semantic
mutations, and lower update/retrieval inconsistency than the strongest baseline
without target-label leakage.

**Kill condition:** the same model generates and judges transformations without
an independent check, or preservation succeeds only on templated paraphrases.

**Ceiling:** at most 5,000 claims and 20 transformations per claim, processed
in bounded batches under 768 MiB RSS and eight minutes per seed.

### `PRW-DDF1` — deference versus deflection

**Purpose:** test whether decision routing and propagation attenuation provide
different, complementary value.

**Dependencies:** `PRW-EK2`.

**Fixture:** sources with unequal reliability, copy clusters, common-mode
failures, contradictions, and query-dependent verification costs.

**Controls:** confidence thresholding, calibrated selective deference,
dependency deflection, both operators, calibrated abstention plus explicit
source-cluster collapse, random attenuation, and oracle lineage as an upper
bound.

**Primary metrics:** selective risk/coverage, cost-adjusted error, false
corroboration, calibration, and independent-evidence recall.

**Null:** separating routing from propagation offers no improvement over
calibrated abstention plus source-cluster collapse.

**Pass boundary:** the joint operator improves the protocol-declared
cost-adjusted-error endpoint on held-out graphs without worse selective risk at
the matched coverage, and each component has a nonzero ablation contribution.

**Kill condition:** one gate creates and validates its own signal, or the joint
method adds no held-out value beyond its strongest component.

**Ceiling:** at most 5,000 nodes and 50,000 edges, under 768 MiB RSS, under five
minutes per seed.

### `PRW-RCM1` — propagation, culling, and operator order

**Purpose:** test whether a bounded recurrent mask offers anything beyond
ordinary diffusion followed by top-\(k\).

**Dependencies:** `PRW-EK2` and `PRW-BIL1`.

**Fixture:** stochastic-block graphs containing true low-degree bridge facts,
irrelevant hubs, a cyclic false-information cluster, contested claims, and
unknown claims.

**Controls:** PageRank/heat diffusion plus top-\(k\) over a pre-collapsed
genealogy, one-shot top-\(k\), scalar walk aggregation,
lineage-set/idempotent propagation, reverberate-then-cull,
cull-then-reverberate, interleaving, random masks, and one-channel versus the
`PRW-BIL1` support/refutation state.

**Primary metrics:** recall at budget, bridge retention, false-cluster
amplification, contested/unknown discrimination, oscillation, convergence,
seed stability, operations, and peak RSS.

**Null:** no recurrent or reordered mask beats ordinary diffusion plus
top-\(k\) over a pre-collapsed genealogy on held-out graphs.

**Pass boundary:** improve the protocol-declared
recall-at-budget/false-amplification frontier over every control, retain all
frozen critical bridge facts, count no cyclic lineage twice, and satisfy the
declared convergence/cycle policy.

**Kill condition:** cyclic walks increase epistemic confidence, contested
claims are removed by scalar cancellation, or adaptive masks fail the declared
convergence/cycle policy.

**Ceiling:** \(n\leq5{,}000\), \(|E|\leq50{,}000\), \(K\leq8\), under 768 MiB
RSS, under five minutes per seed.

### `PRW-RCM1-NLN` — query-reset nonlinear attenuation sidecar

**Purpose:** test whether a passive, locally attached nonlinear auxiliary state
adds a useful state-dependent attenuation mechanism beyond matched adaptive
graph filters and linear sidecars.

**Dependencies:** `PRW-RCM1` and `PRW-ISI1` for candidate-survival work.
The dependency-light Stage-A mathematical sanity protocol may run independently
because it makes no graph-RAG, evidence, or novelty claim.

**Implementation status:** `PROTOCOL_FROZEN_STAGE_A; NOT_RUN`. Source,
formalism, exact first-run fixture, and prior-art collision audit:
`docs/research/nonlinear-neutraliser-subspace-transfer-2026-08.md`.

**Fixture:** Stage A uses frozen scalar Duffing and four-/five-node grounded
path systems. Stage B, if authorized by predecessor dispositions, must use the
`PRW-RCM1` copied-false-cluster/critical-bridge fixture and the paired
protected-versus-nuisance contract from `PRW-ISI1`.

**Controls:** no sidecar, ordinary diffusion/PPR plus top-k, query-conditioned
edge damping/masking, a linear sidecar, GraphCON-style second-order dynamics,
GRAND/GREAD-style nonlinear diffusion, ARMA/GRAMA-style recursive filtering,
one equal-total-mass sidecar, and an equal-state-count co-located two-mode
control.

**Hypothesis:** on held-out evidence graphs and a preregistered amplitude,
frequency, reflection, and propagation-depth grid, a SELECT-chosen query-reset
passive nonlinear sidecar improves the declared nuisance-transmission or
false-cluster-amplification frontier over every matched control while
preserving critical bridge evidence, without unresolved multistability and
within the declared operation/state budget.

**Primary metrics:** linear-limit error, maximum positive unforced energy
increment, hardening peak shift, protected-coordinate leakage, forward/reverse
sweep discrepancy, harmonic content, transmission amplitude, critical-bridge
recall, false-cluster amplification, branch count, operations, state bytes,
p95 latency, and process-tree RSS.

**Null:** after equalizing observations, attachment/certificate maps,
parameters, state bytes, propagation steps, and tuning budget, the nonlinear
sidecar adds no held-out value beyond the strongest matched control.

**Stage-A pass boundary:** satisfy the frozen numerical tolerances, reproduce
the upward hardening-response shift, retain every frequency/branch/failure
cell, keep the protected channel decoupled, and report two distributed
attachments against both equal-total-mass and equal-state-count controls. A
pass validates only the harness and abstraction.

**Candidate-survival boundary:** a future frozen REPORT set must show nonzero
corrected endpoint improvement, no worse selective risk at matched coverage,
at most one percentage point critical-bridge recall loss, no duplicated cyclic
lineage, bounded discrete energy, no unresolved ranking branch, at most 1.5
times the strongest control's operations/p95, and at most 2 times its
auxiliary-state bytes.

**Kill condition:** any matched linear/adaptive control ties or wins; benefit
uses REPORT labels or an oracle nuisance map; protected evidence leaks;
improvement exists only in a selected frequency/amplitude cell; two sidecars
lose to a matched one-sidecar or co-located control; smaller-step or higher-
harmonic checks reverse the result; multistability lacks a label-free branch
policy; or query-local state mutates the evidence ledger.

**Ceiling:** Stage A uses at most five host nodes, two sidecars, 37 frequency
cells, 256 MiB process-tree RSS, and two minutes per deterministic run. Parent
Stage-B/C ceilings remain unopened.

### `PRW-ISI1` — nuisance invariance versus intervention sensitivity

**Purpose:** test whether deflecting nuisance variation preserves sensitivity
to material changes.

**Dependencies:** `PRW-EK3`.

**Fixture:** paired meaning-preserving transformations and material
state-changing interventions for every source claim.

**Controls:** no projection, nuisance-subspace projection, learned deflection,
and certificate-gated deflection.

**Primary metrics:** nuisance violation, missed intervention, false
intervention, and a protocol-declared balanced joint score.

**Null:** deflection cannot improve nuisance invariance without sacrificing
material-change sensitivity.

**Pass boundary:** improve the protocol-declared balanced joint endpoint on the
independent confirmation generator while detecting every frozen
safety-critical intervention.

**Kill condition:** transformation and evaluator generators are not
independent, or one side of the paired endpoint is optimized away.

**Ceiling:** at most 5,000 paired claims, local \(d\leq64\), under 768 MiB RSS,
under five minutes per seed.

### `PRW-OBS1` — bounded off-span observability guard

**Purpose:** distinguish a bounded search region from rank-deficient proposal
support and test whether a valuable direction outside the initial effective
span can be proposed and conditionally qualified.

**Dependencies:** none.

**Implementation status:** `IMPLEMENTED; SYNTHETIC_SANITY_VALIDATED`. Existing CRSV principal-angle
diagnostics and fixed-dimensional Procrustes transforms are predecessor
controls; they do not expand proposal support and are not evidence for this
card. Artifact: `artifacts/method-dev/rb14-observability/obs1.json`.

**Fixture:** use \(d\leq64\), an initial rank \(k\leq8\), equal bounded probe
radius, and response functions that agree on the initial span but differ along
a planted orthogonal direction. Include a narrow valuable-cap case, multiple
separated valuable regions, and a negative control in which the declared
hypothesis generator cannot express the planted direction. The negative
control must remain `UNIDENTIFIED`, not be scored as a discovery failure or a
negative effect.

**Controls:** exploit-only selection, scalar temporal decay, momentum from
in-span gradients, in-span perturbation, bounded isotropic full-rank probes,
optimized periodic persistent excitation, ASEBO-style active/orthogonal
allocation, and BAxUS-style nested expansion. All receive the same response,
probe, radius, and candidate-evaluation budgets.

**Primary metrics:** effective rank, design eigenvalues, span growth, time to
first proposal, time to validated discovery, miss probability, false
qualification, cumulative/simple regret, probe cost, and peak RSS.

**Null:** a support/rank-aware frontier policy provides no discovery or cost
advantage over the strongest bounded full-rank or established subspace-
expansion control.

**Required sanity boundary:** every policy whose proposal support remains in
the initial span is observationally unable to distinguish the paired response
functions; at least one bounded full-rank control obtains positive design
energy in the planted direction. Failure of either condition rejects the
harness rather than supporting a mechanism.

**Candidate survival boundary:** on a separately seeded confirmation set, the
named frontier policy must improve the future protocol's frozen discovery-
delay/miss-cost endpoint over every information- and probe-matched control.

**Kill/route condition:** if isotropic, periodic, ASEBO-style, or BAxUS-style
exploration matches the candidate at the same budget, retain the established
method and drop a distinct open-span mechanism claim. Never infer value from
orthogonality alone; promotion requires an outcome contrast.

**Ceiling:** \(d\leq64\), at most 20,000 bounded probes, one process, under
512 MiB RSS, under five minutes per seed.

### `PRW-COA1` — top-k coalitional-observability ceiling

**Purpose:** test whether a probe policy that observes only singletons or
small masks is exactly blind to recoverable pair/triad effects in the lifted
interaction space.

**Dependencies:** none.

**Implementation status:**
`EXISTING_MOBIUS_DIAGNOSTIC_REUSED; SYNTHETIC_SANITY_VALIDATED`.
The completed `PRW-G2` result—necessary pair/triad residuals validated and
static advantage false—remains unchanged and is a predecessor, not an RB-14
result. Artifact: `artifacts/method-dev/rb14-observability/coa1.json`.

**Fixture:** sparse degree-two and degree-three Boolean response surfaces with
zero or negative singleton effects, planted positive and negative coalitions,
null coalitions, distractors, and a sweep of probe limits
\(\lVert z_t\rVert_0\in\{1,2,3\}\). Include both heredity-respecting and pure
high-order effects so a method is not rewarded only for a favorable hierarchy
assumption.

**Controls:** singleton PCHO-style revisit scheduling, uniform random pairs or
triads, balanced factorial masks, D-optimal/log-determinant designs, sparse
polynomial/combinatorial-bandit estimators, graph-local candidate coalitions,
and an exhaustive oracle reported only as an upper bound. Reuse the existing
Möbius diagnostic to measure interaction; do not present it as a probe policy.

**Primary metrics:** lifted-design rank/eigenvalues, interaction-estimation
error, discovery probability and delay, false promotion, missed coalition,
regret/utility, probe count, operations, and peak RSS.

**Null:** the proposed adaptive coalition scheduler offers no advantage over
the strongest factorial, sparse-polynomial, or combinatorial control at
matched observations and work.

**Required sanity boundary:** when top-k is below the planted interaction
degree, its corresponding design columns are identically zero and recovery
must remain impossible. When eligible joint masks receive adequate support,
the exact small-instance estimator must recover effects above the frozen
minimum size within its declared confidence bound.

**Candidate survival boundary:** the named scheduler must improve the future
protocol's frozen discovery-delay/false-promotion/probe-cost endpoint on a
disjoint confirmation set over every matched non-oracle control.

**Kill/route condition:** if standard factorial, group-testing, sparse-
polynomial, or combinatorial-bandit controls match the result, retain them and
drop a separate coalitional-observability algorithm claim. A nonzero Möbius
residual alone is not a utility win.

**Ceiling:** \(d\leq32\), interaction order \(q\leq3\), at most 20,000 probes,
one process, under 512 MiB RSS, under five minutes per seed.

### `PRW-CTX1` — provenance-compatible joint-context qualification

**Purpose:** test the quantifier failure in which every component has a
favorable witness context but no one compatible context supports their joint
composition.

**Dependencies:** `PRW-COA1` and `PRW-EK2`.

**Implementation status:** `IMPLEMENTED; SYNTHETIC_SANITY_VALIDATED`. Existing CRSV wrong-domain and
order diagnostics, representation-compatibility rejection, and source-lineage
machinery are reused controls; none is a prospective context-conditioned
coalition result. Artifact: `artifacts/method-dev/rb14-observability/ctx1.json`.

**Fixture:** create branches that are individually beneficial in disjoint
contexts or model versions but harmful in every shared context, alongside
valid context-stable positive coalitions, Simpson-style aggregation traps,
adaptive selection, copied lineages, and missing-propensity cases.

**Controls:** naive addition of favorable singleton effects, pooled regression,
context stratification without provenance, joint factorial contrasts with
logged propensities, inverse-propensity and doubly robust qualification, the
same methods after lineage collapse, and an oracle compatible-context
partition reported only as an upper bound.

**Primary metrics:** false compositional promotion, valid-coalition recall,
interaction-effect error, calibration, abstention/`UNIDENTIFIED` rate,
propensity support, probe cost, and operations.

**Null:** immutable context/provenance compatibility adds no protection or
qualified-recall advantage over the strongest correctly stratified,
propensity-aware causal baseline.

**Pass boundary:** reject every frozen incompatible-witness composition,
recover replicated eligible effects above the future protocol's frozen minimum
size, and improve the confirmation-set false-promotion/qualified-recall
frontier over every non-oracle control without using target or oracle-context
labels.

**Kill condition:** the effect disappears after ordinary context
stratification and propensity correction, the tested method constructs and
judges its own contexts, or missing support is silently reported as a negative
effect.

**Ceiling:** at most 5,000 contextual decisions and 20,000 joint probes, one
process, under 768 MiB RSS, under five minutes per seed.

### `PRW-SPU0` — fixed-stack flattening guard

**Purpose:** prevent the closed JO1 mechanism from returning under new
terminology.

**Dependencies:** none.

**Method:** generate fixed projection stacks and compare outputs, distances,
and rank with the explicitly multiplied dense flat map. Separately compare
operations and bytes against both that dense form and the best admissible
factorized-flat form, including lazy ordered application of the same factors.

**Expected result:** output, distance, and rank equality within the declared
arithmetic contract. Operations and bytes are reported as a resource frontier,
not expected to be equal. This is an expressivity guard, not a candidate
benefit.

**Route condition:** any mismatch must first be classified as arithmetic,
state dependence, nonlinearity, gating, or an implementation error. It is not
called dimensional novelty. A win over dense multiplication that is matched by
the factorized-flat control is ordinary factorization, not subplane novelty.

**Ceiling:** \(d\leq128\), \(K\leq8\), under 256 MiB RSS, under two minutes.

### `PRW-SPU1` — heterogeneous subplane unification

**Purpose:** test whether explicit compatibility maps help when local views
really have different dimensions, missingness, and coordinate systems.

**Fixture:** multi-view data with controlled principal angles, variable local
dimensions, missing views, coordinate rotations, view-specific nuisance, and
known compatible/incompatible overlaps.

**Candidate methods:** fiber-product consensus and cellular-sheaf consistency.
Each candidate receives its own contrast and disposition.

**Controls:** padded concatenation/PCA, GCCA, Procrustes, a matched
mixture-of-experts, dense and factorized-flat forms, and shuffled/wrong
compatibility maps. All receive identical observations and relation metadata.

**Primary metrics:** retrieval, false merges, false splits, calibration,
hubness, rank/condition, bytes, operations, and peak RSS.

**Null:** each candidate offers no gain over the strongest flat or established
multi-view baseline at matched information and resources. If both candidates
are carried into confirmation, their two primary contrasts use a Holm
correction frozen in the future protocol.

**Pass boundary:** the named candidate improves its corrected, protocol-declared
retrieval/false-merge/false-split frontier on held-out view structures while
remaining within the matched byte and operation budget; shuffled compatibility
maps must lose the effect.

**Kill condition:** a fixed flat map reproduces the result, wrong maps work
equally well, or the gain comes from extra relation labels denied to controls.

**Dependencies:** `PRW-SPU0`; reuse rather than duplicate the existing CRSV
principal-angle diagnostic.

**Ceiling:** \(n\leq5{,}000\), \(|E|\leq50{,}000\), local \(d\leq64\), under
768 MiB RSS, under eight minutes per seed.

#### `PRW-SPU1-TRANSPORT` — temporal coordinate-transport confound subcell

**Purpose:** determine whether raw distance across representation checkpoints
mistakes a rotation, permutation, bridge change, or rank-preserving basis
change for semantic novelty, forgetting, or atrophy.

**Dependencies:** `PRW-SPU1`. This is a confound-control subcell, not a generic
representation-drift card or an additional architecture component.

**Implementation status:** `IMPLEMENTED; SYNTHETIC_CONFOUND_SANITY_VALIDATED`. Query-encoder
swap, learned realignment, Procrustes, representation bridges, and historical
drift campaigns exist. Their legacy quantitative results remain
`LEGACY_METRIC_LINEAGE_BLOCKED` and are not validation for this subcell.
Artifact: `artifacts/method-dev/rb14-observability/transport.json`.

**Fixture:** hold latent semantics and utility fixed while applying known
rotations, permutations, and versioned bridge maps; separately apply genuine
semantic changes of matched raw magnitude. Fit transport only on training
anchors and evaluate false novelty/forgetting and missed material drift on
held-out anchors, queries, and documents.

**Controls:** no alignment, raw cosine comparison, orthogonal Procrustes,
GCCA/Grassmann or flag-style alignment where dimensions vary, a learned bridge
with identical anchors, shuffled/wrong transport, and oracle transport as an
upper bound.

**Primary metrics:** false novelty, false forgetting, missed semantic drift,
post-alignment residual, retrieval parity, condition number, anchor cost,
bytes, operations, and peak RSS.

**Null:** transport alignment does not reduce coordinate-induced false
novelty/forgetting without also hiding genuine semantic change.

**Pass boundary:** reduce the future protocol's frozen balanced false-
novelty/missed-drift endpoint over raw comparison on held-out structures;
shuffled transport must lose the effect, and oracle transport must recover the
known invariant cases.

**Kill condition:** train/evaluation anchors leak, an unconstrained bridge
changes semantic content, raw comparison already separates coordinate and
semantic changes, or the gain is only a relabeling unsupported by downstream
parity.

**Ceiling:** at most 5,000 vectors per checkpoint, local \(d\leq64\), bounded
anchor batches, one process, under 768 MiB RSS, under five minutes per seed.

### `PRW-VAR1` — Euler-Lagrange relaxation audit

**Purpose:** measure whether a smooth variational relaxation is an adequate
optimizer for the discrete culling problem.

**Dependencies:** none.

**Fixture:** for \(n\leq20\), solve the binary mask problem exactly. Compare the
exact solution with smooth relaxation plus rounding, proximal graph-TV, graph
cuts where applicable, and submodular greedy.

**Primary metrics:** objective/integrality gap, selection Jaccard, feasibility,
downstream recall, runtime, and seed variance.

**Null:** smooth relaxation plus rounding recovers an equivalent feasible mask
and downstream result.

**Kill/route condition:** if rejected, remove Euler-Lagrange as the hard-mask
optimizer and retain it only for a declared smooth state subproblem. Do not
“tune” away an integrality gap.

**Ceiling:** \(n\leq20\) for exact enumeration/MILP and \(n\leq5{,}000\) only
after method selection, under 768 MiB RSS, under eight minutes per cell.

### `PRW-REV1` — reversible culling under correction

**Purpose:** test whether ephemeral masks over an immutable ledger recover
better from delayed correction than destructive or monotone culling.

**Fixture:** streaming observations, retractions, delayed corrections,
coordinate-map changes, and weak bridge facts that later become relevant.

**Controls:** destructive deletion, monotone culling, recomputed one-shot
masks, and ephemeral versioned masks over an append-only ledger.

**Primary metrics:** current and historical recall, reactivation, correction
latency, repair blast radius, storage, and query cost.

**Null:** reversibility and map versioning do not improve correction recovery.

**Pass boundary:** exact historical-view reconstruction, no loss in current
recall, lower correction latency or repair blast radius than the strongest
nondestructive control, and declared storage/query-cost bounds.

**Kill condition:** the ledger cannot reproduce prior views or “reactivation”
depends on evidence retained only by the tested method and denied to controls.

**Dependencies:** `PRW-EK1` and `PRW-RCM1`.

**Ceiling:** at most 5,000 claims and 25,000 events, under 768 MiB RSS, under
five minutes per seed.

### `PRW-EK7` — composite agent/brain memory pilot

**Purpose:** test the surviving pieces together on a bounded temporal RAG/agent
task.

**Entry gate:** do not run unless `PRW-EK0` through `PRW-EK6`, `PRW-BIL1`,
`PRW-DDF1`, `PRW-RCM1`, `PRW-ISI1`, `PRW-SPU0`, `PRW-SPU1`, `PRW-VAR1`,
and `PRW-REV1` each have an explicit disposition. A card becomes an implemented
pilot component only if it passed its eventual frozen independent-confirmation
boundary and has a nonzero ablation contribution. Characterization, reduction,
and method-selection guards remain prerequisites, not features. Do not force
the whole proposed architecture into the pilot.

**Controls:** ordinary vector RAG, recency/version-aware RAG, a Donto-like
bitemporal/provenance baseline, a Donto-like baseline plus copy-aware
corroboration and truth-maintenance repair, a flat composite index with
identical metadata, and ablations for every surviving component.

**Primary metrics:** answer accuracy, current/as-of accuracy, contradiction
handling, source independence, correction recovery, unsafe-action rate,
selective risk/coverage, latency, operations, storage, and process-tree RSS.

**Null:** the surviving integration offers no reproducible advantage over the
strongest established composite baseline at matched evidence and resources.

**Pass boundary:** at least 25% relative reduction in the protocol-declared
false/stale-admission or unsafe-action endpoint, no more than two percentage
points of clean-recall loss, at most 1.5x p95 latency, and at most 2x auxiliary
index storage versus the strongest composite baseline.

**Kill condition:** no component has an independent ablation contribution,
the result relies on hidden metadata/oracle provenance, or the benefit vanishes
under matched retrieval candidates and model calls.

**Ceiling:** frozen small model or deterministic surrogate, capped corpus and
candidate set, one process at a time, under 1 GiB RSS. A separate preflight and
checkpoint/kill-resume test is mandatory before this card.

#### `PRW-EK7-COALITION` — matched-token coalition-RAG survivor ablation

**Purpose:** test whether scoring and retrieving complementary document sets
recovers conjunctive answers that individual-document top-k ranking misses.

**Entry gate:** optional and `NOT_RUN`. It may enter a future `PRW-EK7` protocol
only after `PRW-COA1` passes its required sanity boundary, `PRW-CTX1` has an
explicit disposition defining the admissible context-qualification control,
and the unchanged `PRW-EK7` entry gate is satisfied. The future protocol must
select the established or candidate coalition method before EK7 REPORT outcomes
are visible. This subcell is not a prerequisite for an EK7 pilot that makes no
coalition claim.

**Implementation status:**
`EXISTING_RECURSIVE_DECOMPOSITION_REUSED; MATCHED_TOKEN_SURROGATE_SANITY_VALIDATED`.
The current recursive retriever fuses scores for individual documents; it does
not qualify set-valued complementary evidence. The bounded surrogate is not an
EK7 result and remains entry-gated. Artifact:
`artifacts/method-dev/rb14-observability/coalition_rag.json`.

**Fixture:** questions whose answer requires two or three individually
low-scoring documents, single-document controls, misleading high-scoring
decoys, incompatible-context document sets, copied-source sets, and explicit
unanswerable cases.

**Controls:** ordinary dense top-k, diversified/MMR retrieval, the existing
recursive decomposition with RRF/union/intersection, a strong multi-hop
retriever, graph retrieval with identical relation metadata, random document
sets, and exhaustive set search as a small-instance upper bound. Match corpus,
candidate count, retrieved tokens, model calls, generator, and latency budget.

**Primary metrics:** evidence-set F1, conjunctive Recall@k, answer accuracy,
false coalition promotion, provenance/context violations, abstention,
retrieved tokens, calls, latency, operations, storage, and peak RSS.

**Null:** coalition scoring provides no reproducible benefit over the strongest
multi-hop or graph-retrieval control at matched tokens, calls, metadata, and
candidate opportunity.

**Pass boundary:** improve the future protocol's frozen conjunctive-recall/
answer-accuracy endpoint on a disjoint confirmation set without worse false
coalition or provenance violations and while satisfying all matched-resource
constraints.

**Kill condition:** ordinary recursive, multi-hop, diversified, or graph
retrieval matches the effect; candidate sets or metadata are unmatched; or the
judge sees provenance or answer labels unavailable to controls.

**Ceiling:** deterministic surrogate first, then at most one frozen small
model if authorized; capped corpus and candidate sets, one process, under
1 GiB RSS with the existing EK7 preflight and checkpoint guard.

## 5. Execution order and resource contract

The preservation/publication resume order in `docs/next-session.md` remains
higher priority. RB-13 does not authorize a push, PR mutation, merge, cleanup,
large model load, or full campaign.

### Wave 0A — existing characterization and algebraic guards

1. `PRW-EK0`
2. `PRW-BIL1`
3. `PRW-SPU0`
4. `PRW-VAR1` exact small-instance comparison

### Wave 0B — RB-14 observability guards (bounded sanity complete; survival open)

1. `PRW-OBS1` invariant-span/full-rank guard.
2. `PRW-COA1` top-k interaction-observability guard.

The bounded synthetic harness and exact sanity cells now exist for Wave 0B.
This does not freeze candidate-survival protocols or authorize a production,
model, or corpus run; those still require the independent confirmation and
resource review described on each card.

### Wave 1 — independently falsifiable synthetic mechanisms

1. `PRW-EK1` after `PRW-EK0`.
2. `PRW-EK2` and `PRW-EK3` after `PRW-EK1`.
3. `PRW-EK4` after `PRW-EK3` and `PRW-BIL1`.
4. `PRW-EK5` after `PRW-EK1`, `PRW-EK2`, and `PRW-BIL1`.
5. `PRW-EK6` after `PRW-EK3`.
6. `PRW-DDF1` after `PRW-EK2`.
7. `PRW-RCM1` after `PRW-EK2` and `PRW-BIL1`.
8. `PRW-ISI1` after `PRW-EK3`.
9. `PRW-RCM1-NLN` after `PRW-RCM1` and `PRW-ISI1`; its frozen Stage-A
   mathematical sanity is an independently runnable preflight, not a parent
   disposition.
10. `PRW-SPU1` after `PRW-SPU0`.
11. `PRW-CTX1` after `PRW-COA1` and `PRW-EK2`.
12. `PRW-SPU1-TRANSPORT` after `PRW-SPU1`, only as a confound subcell.
13. `PRW-REV1` after `PRW-EK1` and `PRW-RCM1`.

### Wave 2 — survivor-only integration

1. Freeze each Wave 0/1 disposition and selected implementation hash.
2. Preregister the `PRW-EK7` composite and all ablations.
3. Run the bounded pilot only after the process-tree RSS and
   checkpoint/kill-resume guard is green.
4. Stop at a negative or ordinary-method result; do not scale in search of a
   favorable cell.
5. Add `PRW-EK7-COALITION` only if its entry gate is satisfied; it remains an
   optional ablation and cannot promote unrelated EK7 components.

Every stochastic card uses a deterministic scout grid followed by a separately
seeded confirmation set. All grid cells and all failures are reported. A
scout-selected best cell cannot itself establish a positive result. Ties use a
predeclared deterministic rule. Evidence in an ignored path is not durable
evidence.

The common maximums are:

- \(n\leq5{,}000\), \(|E|\leq50{,}000\), local \(d\leq64\), and propagation
  horizon \(K\leq8\);
- sequential execution, never concurrent experiment processes;
- default process-tree RSS stop at 768 MiB; `PRW-EK7` alone may request 1 GiB
  after a separate preflight;
- no GPU requirement and no model download in Waves 0 or 1;
- child-process timeout, atomic checkpoint, and partial-artifact quarantine;
- immutable input/config hashes and a content-bound manifest for retained
  evidence.

The RSS stop is a reactive process-tree guard, not a kernel-enforced allocation
limit. Every card must also estimate aggregate allocations before launch and
refuse the run when the modeled peak exceeds its ceiling. A reactive monitor
alone is not an OOM guarantee.

## 6. Promotion and reporting boundary

A positive synthetic result establishes only behavior on the frozen fixture.
It does not establish truth, consciousness, a brain substrate, lower training
cost, production utility, or novelty.

For each card:

1. Freeze the operator, baselines, endpoint, resource accounting, and kill
   criteria before seeing REPORT outcomes.
2. Report the full scout grid, then a disjoint confirmation set.
3. Separate statistical uncertainty from fixture and evaluator validity.
4. Charge all relation labels, provenance edges, certificate checks, storage,
   model calls, and preprocessing to the tested method.
5. Compare against the strongest established control, not only a no-op.
6. Preserve negative, null, and category-error results.
7. Require independent literature and proof review before using `novel`.

The narrow candidate worth testing is not “dimensional stacking.” It is:

> A bitemporal, provenance-carrying evidence kernel whose epistemic state is
> lattice-valued; whose language, embedding, and heterogeneous local-space
> representations are reversible/versioned views; and whose query-time
> propagation and culling are source-idempotent, correction-reversible, and
> separated from action authority.

That integration is still a conjecture. The queue is designed to discover
whether it reduces to ordinary memory, graph filtering, multi-view learning,
or selective prediction before a larger system is built.

## 7. Cross-cutting failure modes

The cards also need these system-level guards:

- **Unknown genealogy:** missed copy/common-cause edges can turn an
  independence certificate into overconfidence. Exact genealogy and inferred
  genealogy must be reported separately.
- **Cycles:** contradiction and derivation graphs are not guaranteed DAGs.
  Local repair must condense strongly connected components or use a declared
  fixed-point policy before propagation.
- **Claim typing as oracle:** claim-type labels and obligations can leak the
  generator or target. Typing error must be measured separately.
- **Shared evaluators:** a renderer, parser, verifier, and judge built from the
  same model can share one defect. Safety-critical gates need independent or
  deliberately diverse checks.
- **Time-of-check/time-of-use:** certificates, claim versions, policies,
  identity lenses, and action identities must be bound atomically. A later
  policy or evidence change invalidates the old decision without deleting it.
- **Privacy and erasure:** an append-only research ledger can conflict with
  deletion, consent, and retention obligations. Access revocation, redaction,
  cryptographic erasure, and audit-preserving tombstones require a separate
  policy design before real personal data.
- **Formal/empirical boundary:** a proof certifies its specification and
  assumptions, not the external world represented by them.
- **Resource leakage:** lineage edges, compatibility maps, preprocessing,
  verifier calls, and historical indexes are charged to the tested method.

## 8. Prior-art anchors

These anchors bound, rather than establish, novelty:

- Donto public systems report (public prior art, not independent peer-reviewed
  validation):
  <https://donto.org/reports/donto-paper-2026-05-28>
- Mozannar and Sontag, learning to defer:
  <https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf>
- Hansen and Ghrist, cellular sheaf Laplacians:
  <https://arxiv.org/abs/1808.01513>
- Ghrist and Riess, Tarski Laplacian:
  <https://arxiv.org/abs/2007.04099>
- Dong et al., source-dependency-aware truth discovery:
  <https://www.vldb.org/pvldb/vol2/vldb09-pvldb47.pdf>

The reference list is a starting boundary, not a completed patent,
publication, or exhaustive prior-art search.
