# Nonlinear-neutraliser transfer audit for graph propagation and RAG

**Audit ID:** `RB-15`

**Date:** 2026-08-12

**Status:** source review complete enough to freeze a bounded mathematical
sanity test; scientific, product, and novelty claims remain unconfirmed

**Candidate subcell:** `PRW-RCM1-NLN`

## 1. Bottom line

Frances Fulton's 2026 presentation contributes a useful mechanism, but not the
broad mechanism that the word *subspace attenuation* might suggest.

The transferable object is an **attached, local, stateful, nonlinear filter**.
It changes its effective response with the amplitude and phase that reach it.
Several attachments compose as an ordered feedback system because every
upstream scattering event changes the forcing seen downstream, and reflections
can return that changed state upstream.

That is a rigorous reinterpretation of several CHELATEDAI metaphors:

- an "onion layer" becomes an ordered stateful transfer stage;
- "resonance" becomes a measured amplitude-dependent frequency response;
- "reverberation" becomes bounded propagation with delayed feedback;
- "living memory" becomes a resettable auxiliary state, not an authoritative
  truth store;
- "orthogonal additives" become explicit protected and nuisance maps.

It does **not** create new observable directions, discover unknown positive
vectors, establish higher-dimensional geometry, make repeated sources
independent, or make stored claims true. Those remain separate problems.

## 2. Primary source chain and chronology

1. [Fulton, *Wave Manipulation in Structures with Attached Nonlinear
   Neutralisers*](https://www.youtube.com/watch?v=eQwG_xl-h_k), Isaac Newton
   Institute seminar, 11 August 2026. The 29:54 talk presents current PhD work,
   centered on two Duffing-type neutralisers attached to an axially vibrating
   rod.
2. Fulton, Sorokin, and Abdi,
   [*Elastic wave transmission through a semi-infinite rod with an attached
   damped nonlinear neutraliser*](https://doi.org/10.1007/s11071-025-11658-3),
   *Nonlinear Dynamics* 113 (2025), 28657-28673. This is the peer-reviewed
   single-neutraliser mathematical source.
3. Fulton,
   [*Nonlinear periodically attached absorbers for vibration transmission
   mitigation in linear structures*](https://www.otago.ac.nz/__data/assets/pdf_file/0021/556131/KOZWaves-2024-conference-booklet.pdf),
   KOZWaves 2024, pp. 15-16. This record describes completed single-absorber
   work and preliminary two-absorber work.
4. Sorokin, Fulton, and Thomsen,
   [*Theoretical and Experimental Analysis of Axial Wave Transmission in a Rod
   with a Nonlinear Absorber*](https://esmc2025.sciencesconf.org/593130/document),
   ESMC 2025. This record describes the Duffing model and the single-absorber
   experiment.
5. Sorokin et al.,
   [*Theoretical and experimental analysis of axial wave transmission in a rod
   with attached nonlinear absorbers*](https://www.kozwaves2026.org/booklet_050126.pdf),
   KOZWaves 2026, p. 28. This adds a reported autoparametric-pendulum line and
   experimental comparison with a linear absorber.

No public thesis, source code, dataset, or peer-reviewed two-neutraliser paper
was located in the bounded search. The talk says thesis submission was still
forthcoming. The 2025 paper states that no datasets were generated or analysed.
The multi-neutraliser results must therefore be treated as current conference
work, not independently reproduced evidence.

## 3. What the physical model actually says

### 3.1 Local attachment

Let (u_h(t)) be host displacement at an attachment, (u_z(t)) the attached
mass displacement, and

\[
r(t)=u_z(t)-u_h(t)
\]

the relative displacement. The local attachment force has the Duffing form

\[
f(r,\dot r)=\beta\dot r+k_1r+k_3r^3.
\]

The 2025 paper couples that force to rod continuity and force balance, reduces
the wave problem to an ordinary differential equation in (r), and solves a
harmonic approximation by the method of varying amplitudes. Its declared
dimensionless controls include

\[
\Omega=\frac{\omega}{\omega_n},\qquad
\mu=\frac{m\omega_n}{Z},\qquad
\eta=\frac{\beta}{2\sqrt{k_1m}},\qquad
\gamma=\frac{k_3q_a^2}{k_1},
\]

plus incident-wave amplitude ratio Q, phase difference varphi, and
dimensionless distance (Delta) to a reflecting boundary or attachment.

For a near-harmonic relative motion (r(t)=A\sin\omega t),

\[
r^3=\frac{3}{4}A^3\sin\omega t-\frac{1}{4}A^3\sin3\omega t.
\]

The first-harmonic approximation therefore sees

\[
k_{\mathrm{eff}}(A)=k_1+\frac{3}{4}k_3A^2.
\]

This is the precise source of the amplitude-dependent tuning intuition. A
positive (k_3) hardens and moves the response upward with amplitude; a
negative (k_3) softens it and can lose global coercivity.

### 3.2 Multiple attachments are stateful composition

The 2026 talk treats two attachments separately, relates the waves between
them by phase delay, solves the first attachment initially without a returned
wave, solves the second attachment, returns its reflected wave to the first,
and iterates until a steady state is reached.

Thus a later attachment does not receive the original input. It receives an
amplitude- and phase-modified state. Formally, the total transfer is not a
fixed product (T_2T_1); it is closer to

\[
y=T_2\!\left(\omega,A_2(y),\phi_2(y)\right)
  T_1\!\left(\omega,A_1(y),\phi_1(y)\right)x,
\]

with reflected terms closing a feedback loop. Order, distance, amplitude,
phase, initial condition, and branch-selection policy can all matter.

### 3.3 Reported positives, negatives, and limits

- In the 2025 single-neutraliser cells, hardening nonlinearity can widen and
  shift the declared low-transmission band upward relative to the linear
  control. Softening is less reliable and needs adequate damping.
- Phase and amplitude of a second incident wave can improve or worsen
  transmission. More incident energy increases effective nonlinearity but can
  also raise transmission and add unstable branches.
- Distance to a reflecting boundary can create higher transmission, unstable
  solutions, and isolas. Careful tuning is a condition, not a minor detail.
- The 2026 two-neutraliser result shows possible broadening at a 20% reduction
  threshold, but also loops, isolas, jump behavior, and duplicated solution
  branches under iteration.
- For the displayed parameter cell, varying the separation distance changed
  response complexity but did **not** broaden the transmission dip.
- The two-neutraliser harmonic approximation mainly retains the fundamental.
  The question period explicitly identifies third-harmonic validation as open,
  particularly outside low-to-moderate nonlinearity.
- Numerical integration broadly agrees on stable branches, with local
  deviations. The physical experiment validates only the single neutraliser.

The honest conclusion is therefore a candidate mechanism plus strong failure
tests, not a demonstrated layered attenuation architecture.

## 4. Terminology boundary: neutraliser is not automatically an energy sink

Fulton's attachment has linear stiffness, cubic stiffness, and optional
damping. It is closest to a nonlinear tuned vibration absorber/neutraliser.
The classical nonlinear-energy-sink literature often reserves *energy sink*
for an essentially nonlinear attachment with no linear eigenfrequency and an
activation-dependent targeted-energy-transfer regime.

For this audit:

- (k_3r^3) stores and returns energy through a nonlinear potential;
- the damping term beta times the relative velocity dissipates energy;
- scattering can reflect or redistribute energy;
- nonlinear response can transfer energy to higher harmonics;
- none of those facts alone proves one-way removal of an information error.

Calling the proposed AI mechanism an "energy sink" before defining the energy,
input/output ports, and dissipation inequality would be a category error.

## 5. Formal CHELATEDAI transfer

### 5.1 State and attachment map

For a query (q), let:

- (x(t)\in\mathbb{R}^{nd}) be the host evidence/feature state over a graph;
- (L_q\succeq0) be the fixed query-conditioned propagation operator;
- (C_q\in\mathbb{R}^{m\times nd}) be an outcome-blind attachment map derived
  from declared provenance, dependency, contradiction, or nuisance
  certificates;
- (z(t)\in\mathbb{R}^{m}) be a query-local sidecar state;
- (r(t)=z(t)-C_qx(t)) be the sidecar/host mismatch.

The candidate continuous-time system is

\[
M_x\ddot x+D_x\dot x+L_qx-C_q^\top f(r,\dot r)=s_q(t),
\]

\[
M_z\ddot z+f(r,\dot r)=0,
\]

where

\[
f(r,\dot r)=D_r\dot r+K_rr+\kappa\odot r^{\odot3}.
\]

The sidecar is **not** part of the authoritative evidence ledger. It is reset
for each query or bounded propagation episode. Every output must retain the
ledger nodes, edges, versions, and provenance used to produce it.

### 5.2 Passivity screen

Assume (M_x,M_z\succ0), (D_x,D_r\succeq0), (L_q,K_r\succeq0), and
(kappa_j\geq0). Define

\[
\begin{aligned}
E={}&\frac12\dot x^\top M_x\dot x
   +\frac12x^\top L_qx
   +\frac12\dot z^\top M_z\dot z \\
  &+\frac12r^\top K_rr
   +\sum_j\frac{\kappa_j}{4}r_j^4.
\end{aligned}
\]

Then the equal-and-opposite coupling gives

\[
\dot E=\dot x^\top s_q
       -\dot x^\top D_x\dot x
       -\dot r^\top D_r\dot r.
\]

With no input, (\dot E\leq0). This is the minimum admissibility condition:
the sidecar may store and return state, while declared damping removes energy;
it may not create hidden unbounded gain.

This certificate is conditional. It can fail after discretization, with a
nonsymmetric learned operator, negative cubic stiffness, outcome-dependent
attachment maps, or numerical steps outside the integrator's stability region.
Those are test obligations, not implementation details.

### 5.3 Protected and nuisance components

Let (P_q) select protected evidence and (Q_q) select a candidate nuisance
component. An exact synthetic invariant cell requires

\[
C_qP_q=0.
\]

Real representations will rarely satisfy exact orthogonality, so the live
metric is protected leakage, not a verbal promise that the two spaces are
independent:

\[
\epsilon_P=
\frac{\|P_q(x_{\mathrm{candidate}}-x_{\mathrm{control}})\|_2}
     {\|P_qx_{\mathrm{control}}\|_2+\varepsilon}.
\]

The nuisance map must be built without REPORT labels or answer correctness. A
certificate learned from the same target it later "protects" invalidates the
experiment.

## 6. Exact hypothesis

### `PRW-RCM1-NLN-H1`

On held-out evidence graphs and a preregistered amplitude, frequency,
reflection, and propagation-depth grid, a query-reset passive nonlinear
sidecar selected only on SELECT will improve the declared
nuisance-transmission/false-cluster-amplification frontier over every matched
control while preserving critical bridge evidence, without unresolved
multistability and within the declared operation/state budget.

### Null

After equalizing observations, certificate maps, parameters, state bytes,
propagation steps, and tuning budget, nonlinear sidecars add no held-out value
beyond the strongest of:

1. ordinary diffusion/PPR plus top-(k);
2. query-conditioned edge damping or masking;
3. a linear sidecar ((kappa=0));
4. GraphCON-style second-order primary-state dynamics;
5. GRAND/GREAD-style nonlinear diffusion or reaction-diffusion;
6. ARMA/GRAMA-style recursive state-space filtering;
7. a port-Hamiltonian deep graph network; and
8. a matched compositional port-Hamiltonian auxiliary-state model.

### Candidate-survival boundary

The future frozen protocol must require all of the following on REPORT:

1. a nonzero corrected improvement in the preregistered nuisance-gain or
   false-cluster-amplification endpoint over the strongest control;
2. no worse selective risk at matched coverage;
3. no more than one percentage point loss of frozen critical-bridge recall;
4. no cyclic lineage counted as independent evidence;
5. all unforced trajectories pass the discrete energy/boundedness screen;
6. forward and reverse sweeps plus multiple initial states expose no unresolved
   output branch used by the ranking policy;
7. at most (1.5\times) the strongest control's operations/p95 latency and at
   most (2\times) its auxiliary state bytes.

These are survival gates, not evidence that the method is useful. Practical
effect-size and statistical thresholds must be frozen before any live corpus
campaign.

### Kill conditions

Kill or reduce the candidate if any one holds:

- the linear sidecar, adaptive scalar gate, CatRAG-style query-conditioned
  transition, GraphCON, GREAD, GRAMA, port-Hamiltonian graph, or compositional
  port-Hamiltonian control matches it;
- benefit requires an oracle nuisance map or answer labels;
- the protected channel changes materially despite equal information;
- improvement appears only at a selected amplitude/frequency/phase cell;
- a two-sidecar system is worse than one sidecar or a single matched larger
  sidecar on the preregistered endpoint;
- bidirectional sweeps expose hysteresis, isolas, or multiple stable ranking
  outputs without a label-free branch-selection policy;
- higher-harmonic or smaller-step integration reverses the result;
- the result is ordinary factorization, clipping, or extra state/search budget;
- query-local state leaks into or mutates the authoritative evidence ledger.

## 7. Bounded experiment sequence

### Stage A — mathematical sanity only

Use a small path/mesh graph with scalar host dynamics and one or two attached
sidecars. No model, corpus, GPU, or production retrieval path.

The first execution is frozen before inspecting its outputs:

- float64 arithmetic and fixed-step classical RK4;
- one four-node grounded path for the linear-limit, energy, and protected-
  channel checks, and one five-node grounded path for the attachment-location
  comparison;
- host mass identity, host damping 0.04 times identity, path-Laplacian
  stiffness plus 0.20 times identity, and zero outcome-derived inputs;
- four-node sidecar parameters: attach at zero-based node 1 with mass 0.20,
  linear stiffness 0.20, relative damping 0.032, and cubic stiffness 0.50.
  Start the host at [0.10, -0.05, 0.02, 0.00], the sidecar at 0.03, and all
  velocities at zero; run the exact-linear comparison to time 1.0 and the
  unforced energy trace to time 4.0 with step 0.001. Run 11 multiplies every
  nonzero initial displacement by -0.8;
- for the protected-channel control, duplicate that four-node path into
  block-diagonal nuisance and protected feature channels with identical host
  operators. Point-attach the sidecar only to zero-based node 1 of the nuisance
  channel and compare the protected trajectory with the matched no-sidecar
  reference. This is the exact synthetic construction of an attachment map
  that annihilates the protected feature channel;
- hardening response control: the scalar forced Duffing relative-coordinate
  equation with mass 1, linear stiffness 1, damping 0.12, cubic stiffness 1,
  forcing amplitudes 0.05 and 0.30, and 37 equally spaced frequencies from
  0.70 through 1.60; integrate 60 periods per cell with 200 fixed steps per
  period and measure the last 20 periods in both forward and reverse order;
- graph attachment comparison: 17 equally spaced frequencies from 0.70
  through 1.50 at forcing amplitude 0.20, comparing no sidecar, one
  equal-total-mass sidecar, two distributed half-mass sidecars, and two
  co-located half-mass sidecars. The co-located control matches the two-
  sidecar auxiliary-state count; all physical mass, linear stiffness, damping,
  and cubic coefficient totals are also reported. The single sidecar at
  zero-based node 2 uses totals (mass 0.40, linear stiffness 0.40, relative
  damping 0.064, cubic stiffness 1.00); distributed sidecars at nodes 1 and 3,
  and co-located sidecars at node 2, each split those totals equally. Force
  node 0, measure the fundamental amplitude at node 4, and integrate 60 periods
  per cell at 200 steps per period while measuring the final 20;
- a linear-limit final-state tolerance of 1e-6 against an independently
  eigendecomposed state transition, maximum positive unforced energy-step
  tolerance of 1e-9, and protected-coordinate leakage tolerance of 1e-12;
- deterministic run identifiers 7 and 11 perturb only the declared initial
  displacement sign/scale; neither run may tune a parameter or discard a
  frequency cell. Forward and reverse sweeps are independent continuations
  from an all-zero state at their respective starting endpoint and warm-start
  only their own subsequent cells; neither branch selects the reported result.
  Run 11 changes only the explicitly declared nonzero four-node initial state,
  so the forced grids are intentionally identical across run IDs and must not
  be misreported as an independent robustness replication; and
- a 256 MiB process-tree RSS ceiling and two-minute wall-clock ceiling per run.

If the scalar hardening peak does not move upward, a tolerance fails, or a run
crosses its resource ceiling, record the failure. Do not change the frozen
fixture in the same evidence generation.

Required checks:

1. (kappa=0) agrees with the independently solved linear state-space control;
2. unforced hardening cells have nonincreasing energy within a declared
   integrator tolerance;
3. increasing input amplitude moves the hardening response upward in frequency
   on the frozen fixture;
4. a decoupled protected channel is bitwise or tolerance-identical;
5. two attachments are compared against one equal-total-state attachment;
6. all frequency cells, unstable runs, nonconvergence, and reverse-sweep
   discrepancies are retained.

Passing Stage A validates the implementation and abstraction only. It cannot
support an AI, RAG, or novelty claim.

### Stage B — synthetic evidence graph

Use the `PRW-RCM1` fixture: a true low-degree bridge, irrelevant hubs, a cyclic
copied-false cluster, contested claims, unknown claims, and a two-channel
support/refutation state inherited from `PRW-BIL1`.

The same provenance/dependency information is supplied to all controls. Measure
critical bridge recall, false-cluster amplification, contested/unknown
discrimination, convergence, oscillation, branch count, operations, state
bytes, and process-tree RSS.

### Stage C — survivor-only graph RAG

Only after `PRW-RCM1`, `PRW-ISI1`, and the Stage-B sidecar contrast have
explicit dispositions, compare against ordinary vector RAG, HippoRAG/PPR,
GNN-RAG, query-conditioned traversal, and a matched adaptive state-space graph
filter on a frozen multi-hop corpus. The sidecar state is reset per query and
never changes stored evidence.

## 8. Relationship to existing CHELATEDAI cards

| Card | Relationship | What this source does not establish |
|---|---|---|
| `PRW-RCM1` | Direct optional subcell: stateful local attenuation and operator order | No advantage over diffusion/masks yet |
| `PRW-DDF1` | Sidecar parameters could be a deflection action after a separate routing decision | No calibrated deference policy |
| `PRW-ISI1` | Supplies the required protected-versus-nuisance safety contrast | No valid nuisance projector from the video |
| `PRW-BIL1` | Support/refutation channels prevent scalar cancellation | No epistemic meaning for mechanical displacement |
| `PRW-SPU1` | At most reuses compatibility maps as attachment maps | No heterogeneous-space unification result |
| `PRW-SPU1-TRANSPORT` | Coordinate drift remains a confound | No learned transport or semantic invariance |
| `PRW-OBS1` | None: dynamics stay within the declared reachable state | No off-span discovery |
| `PRW-COA1` | None unless explicit interaction probes are added | No coalitional observability |
| `PRW-REV1` | Query-local reset and immutable history are compatible | No persistent truth or correction ledger |

Prime length 4691, rotation factorization, onion rings, polar coordinates, and
higher embedding dimension receive no new evidence from this source.

## 9. Prior-art collision map

| Work | Occupied territory | Remaining distinction, if any |
|---|---|---|
| [GraphCON (ICML 2022)](https://proceedings.mlr.press/v162/rusch22a.html) | nonlinear controlled damped graph-coupled oscillators; second-order hidden dynamics; oversmoothing and gradient analysis | candidate keeps a separate attached sidecar and targets certified nuisance propagation rather than making the primary node state the oscillator |
| [GRAND (ICML 2021)](https://proceedings.mlr.press/v139/chamberlain21a.html) | linear/nonlinear graph diffusion with numerical stability | candidate uses a coupled auxiliary state and passivity screen, not only a diffusion coefficient |
| [GREAD (ICML 2023)](https://proceedings.mlr.press/v202/choi23a.html) | learned reaction-diffusion graph dynamics | candidate must beat its nonlinear local-reaction control at equal state and tuning budget |
| [Wavy Transformer (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b7aa34d2d24f9bab3056993b7bfa0f1b-Abstract-Conference.html) | second-order wave dynamics for attention and oversmoothing | rules out novelty from "wave-like transformer layers" alone |
| [Stuart-Landau GNN (2025 preprint)](https://arxiv.org/abs/2511.08094) | joint amplitude/phase oscillator dynamics and multistable synchronization | rules out novelty from amplitude-aware graph oscillators alone |
| [ARMA graph filters](https://arxiv.org/abs/1602.04436) | distributed recursive graph filtering with stability conditions | a linear sidecar is largely a state-space/rational-filter realization |
| [GRAMA (ICML 2025)](https://proceedings.mlr.press/v267/eliasof25a.html) | adaptive graph ARMA/state-space propagation with selective coefficients | rules out novelty from adaptive recursive coefficients or auxiliary state alone |
| [Port-Hamiltonian Deep Graph Networks (ICLR 2025)](https://openreview.net/forum?id=03EkqSCKuO) | balances conservative and dissipative information flow in message-passing graphs with energy-based guarantees | rules out novelty from passivity-regulated graph propagation itself and is a mandatory architectural control |
| [Compositional port-Hamiltonian neural networks (L4DC 2023)](https://proceedings.mlr.press/v211/neary23a.html) | composes learned nonlinear spring-mass-damper subsystems through known or learned interconnections while retaining cyclo-passivity | rules out novelty from modular passive mechanical sidecars or nonlinear attachment composition alone |
| [GNN-RAG (ACL 2025)](https://aclanthology.org/2025.findings-acl.856/) | learned query-relevant graph propagation for efficient KG retrieval | mandatory live RAG control |
| [CatRAG (2026 preprint)](https://arxiv.org/abs/2602.01965) | query-conditioned edge reweighting plus PPR to reduce hub drift and retain evidence chains | candidate must beat dynamic graph steering without using more semantic/oracle information |
| [Fang et al. (2017)](https://www.nature.com/articles/s41467-017-00671-9) | experimental nonlinear metamaterial attenuation, chaos, multistate response, amplitude-dependent bands | confirms physical field maturity and makes "nonlinear broadband attenuation" non-novel |

This search did not locate an exact paper combining all of: query-reset
attached sidecars, certificate-limited nuisance coupling, a passive hardening
potential, immutable provenance, and graph-RAG utility/branch gates. Absence
from a bounded search is not proof of novelty. Port-Hamiltonian graph networks
and compositional port-Hamiltonian subsystem learning materially occupy the
closest mathematical architecture; the remaining distinction is evidence- and
retrieval-specific rather than a new dynamics family.

## 10. Decision

Add `PRW-RCM1-NLN` as a **nonpromoted optional subcell**, not as a new theory
family. Authorize only the dependency-light Stage-A sanity harness in the
current pass. Do not alter the production retrieval path and do not claim that
the source solved subspace discovery, semantic truth, RAG recall, training
cost, or computing history.

The source's most valuable contribution to CHELATEDAI is a disciplined form of
state-dependent attenuation plus a list of ways it can fail.
