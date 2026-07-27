# Exact leading-shell proof for the frozen prime-ring bank

**Proof note ID:** `CHELATEDAI-PRW-T1R-PROOF-2026-07`

**Model:** two types, eight layers, `typed16` masks, ideal Legendre carrier

**Claim status:** `PRW-T1R-TE-EVENT-UNION:
CLOSED_INTERNAL_ANALYTIC_SCOPE`

**Canonical decoder status:** `PRW-T1D-CANONICAL-FIRST-TYPE0:
ANALYTIC_TIE_COROLLARY_COMPLETE`

**Type-one decoder status:** `PRW-T1D-CANONICAL-FIRST-TYPE1:
ANALYTIC_TIE_COROLLARY_COMPLETE`

**All-state decoder status:** `PRW-T1D-ALL-STATES:
ANALYTIC_TIE_COROLLARY_COMPLETE_CURRENT_FLOAT_FFT_LINKAGE_FAILED`

**Evidence status:** non-confirmatory method development; no novelty or
production claim

## 1. Exact scope

Fix:

- a prime \(p>7\) with \(p\equiv3\pmod4\);
- the eight layer indices \(V=\mathbb F_2^3\), written as the integers
  \(0,\ldots,7\) when embedded in \(\mathbb Z_p\);
- the two phase signatures
  \[
  \phi_0(x)=0,\qquad \phi_1(x)=x;
  \]
- the bipolar length-\(p\) Legendre carrier \(c\), with
  \[
  \langle c,R_a c\rangle=
  \begin{cases}
  p,&a=0,\\
  -1,&a\ne0;
  \end{cases}
  \]
- the `typed16` mask family
  \[
  \mathcal M
  =
  \left\{
  m_{u,b}(x)=(-1)^{u\cdot x+b}:
  u\in\mathbb F_2^3,\ b\in\mathbb F_2
  \right\};
  \]
- independent binary-symmetric-channel flips with fixed
  \(q\in(0,1/2)\).

The clean bank state indexed by type \(t\), global shift \(s\), and mask \(m\)
is

\[
h_{t,s,m}(x,\cdot)=m(x)R_{\phi_t(x)+s}c.
\]

The proof first fixes the canonical transmitted state

\[
h_*=h_{0,0,\mathbf 1}.
\]

Section 8 explains why this is without loss for the frozen bank. No extension
is claimed for other signatures, mask families, growing layer count,
non-i.i.d. channels, or \(q=q_p\) approaching \(1/2\).

For a wrong-type state \(h_i\), let \(D_i\) be its disagreement set with
\(h_*\), let \(d_i=|D_i|\), and let

\[
E_i^{\ge}
=
\left\{
\text{the channel flips at least }\left\lceil d_i/2\right\rceil
\text{ coordinates of }D_i
\right\}.
\]

This is the event that competitor \(i\) ties or beats the truth in correlation.
The theorem concerns the event union

\[
U_p^{\ge}=\bigcup_{i\text{ wrong type}}E_i^{\ge}.
\]

It is not yet the final class-error event of a decoder with an unspecified tie
rule.

## 2. Two elementary group lemmas

### 2.1 Mask weights and signs

Pointwise multiplication makes \(\mathcal M\) a group isomorphic to
\(\mathbb F_2^4\). Its members have exactly three possible negative weights:

- \(m_{0,0}=\mathbf1\) has weight \(0\);
- \(m_{0,1}=-\mathbf1\) has weight \(8\);
- the fourteen masks with \(u\ne0\) are balanced and have weight \(4\).

For any fixed layer \(a\), the fourteen balanced masks split into seven with
\(m(a)=+1\) and seven with \(m(a)=-1\). This follows by pairing every
nonconstant character \(m\) with \(-m\).

More precisely, the seven balanced masks positive at \(a\) are

\[
m_{a,u}(x)=(-1)^{u\cdot(x+a)},
\qquad
u\in\mathbb F_2^3\setminus\{0\}.
\]

Here addition inside the exponent is in \(\mathbb F_2^3\); phase differences
below use the integer representatives in \(\mathbb Z_p\).

### 2.2 One-layer Hamming distances

For bipolar words \(z,w\) of length \(p\),

\[
d_H(z,w)=\frac{p-\langle z,w\rangle}{2}.
\]

Consider one layer of a candidate with relative phase \(a\) and relative mask
sign \(\epsilon\). The ideal Legendre autocorrelation gives

\[
\begin{array}{c|cc}
&\epsilon=+1&\epsilon=-1\\ \hline
a=0&0&p\\
a\ne0&(p+1)/2&(p-1)/2.
\end{array}
\tag{2.1}
\]

This four-entry table is the only carrier fact needed below.

## 3. Derivation of the seven wrong-type shells

A wrong-type state has relative phase \(x+s\) in layer \(x\). Because
\(p>7\), the eight residues \(0,\ldots,7\) are distinct in \(\mathbb Z_p\).
Consequently:

- for each of eight shifts \(s=-a\), exactly layer \(a\) is aligned;
- the remaining \(p-8\) shifts align no layer.

Let \(k\) be the negative weight of a mask.

### 3.1 Aligned shift and positive aligned sign

If the aligned layer has positive sign, all \(k\) negative signs occur among
the seven unaligned layers. Equation (2.1) gives

\[
d_+(k)
=
\frac{k(p-1)+(7-k)(p+1)}{2}
=
\frac{7p+7-2k}{2}.
\tag{3.1}
\]

At each aligned layer there are seven balanced positive masks and the one
all-positive mask. Thus:

- \(k=4\) gives distance \((7p-1)/2\), with multiplicity \(8\cdot7=56\);
- \(k=0\) gives distance \((7p+7)/2\), with multiplicity \(8\).

### 3.2 Aligned shift and negative aligned sign

If the aligned sign is negative, its layer contributes \(p\), and the other
seven layers contain \(k-1\) negative and \(8-k\) positive signs. Therefore

\[
d_-(k)
=
p+\frac{(k-1)(p-1)+(8-k)(p+1)}{2}
=
\frac{9p+9-2k}{2}.
\tag{3.2}
\]

At each aligned layer there are seven balanced negative masks and the one
all-negative mask. Thus:

- \(k=4\) gives distance \((9p+1)/2\), with multiplicity \(8\cdot7=56\);
- \(k=8\) gives distance \((9p-7)/2\), with multiplicity \(8\).

### 3.3 Unaligned shift

When no layer aligns, equation (2.1) gives

\[
d_0(k)
=
\frac{k(p-1)+(8-k)(p+1)}{2}
=
4p+4-k.
\tag{3.3}
\]

For each of the \(p-8\) unaligned shifts:

- the all-negative mask gives \(4p-4\), with total multiplicity \(p-8\);
- the fourteen balanced masks give \(4p\), with total multiplicity
  \(14(p-8)\);
- the all-positive mask gives \(4p+4\), with total multiplicity \(p-8\).

Combining (3.1)--(3.3) yields the complete spectrum

\[
\begin{array}{c|c|l}
\text{distance}&\text{multiplicity}&\text{source}\\ \hline
(7p-1)/2&56&\text{aligned, balanced, positive at alignment}\\
(7p+7)/2&8&\text{aligned, all positive}\\
4p-4&p-8&\text{unaligned, all negative}\\
4p&14(p-8)&\text{unaligned, balanced}\\
4p+4&p-8&\text{unaligned, all positive}\\
(9p-7)/2&8&\text{aligned, all negative}\\
(9p+1)/2&56&\text{aligned, balanced, negative at alignment}.
\end{array}
\tag{3.4}
\]

The multiplicities sum to

\[
56+8+(p-8)+14(p-8)+(p-8)+8+56=16p,
\]

which is exactly the number of wrong-type shift/mask states.

## 4. Derivation of the five leading overlap classes

Set

\[
d_N=\frac{7p-1}{2},
\qquad
d_A=d_N+4=\frac{7p+7}{2}.
\]

The 64 leading states are:

- the adjacent states \(A_a\), one all-positive state aligned at each
  \(a\in V\);
- the nearest states \(N_{a,u}\), with aligned layer \(a\in V\) and
  \(u\in V\setminus\{0\}\), whose mask is \(m_{a,u}\).

For any two competitors \(i,j\), their disagreement-set intersection obeys

\[
|D_i\cap D_j|
=
\frac{d_i+d_j-d_H(h_i,h_j)}{2}.
\tag{4.1}
\]

If their aligned layers are equal, their carrier phases agree in all eight
layers, so their mutual distance is \(p\) times the negative weight of their
relative mask. If their aligned layers differ, every relative phase is the
same nonzero residue; a relative mask of negative weight \(k\) then gives
mutual distance \(4p+4-k\) by (3.3).

### 4.1 Adjacent--adjacent pairs

Two distinct adjacent states have different aligned layers and relative mask
\(\mathbf1\). Hence their mutual distance is \(4p+4\), and (4.1) gives

\[
|D_{A_a}\cap D_{A_b}|=\frac{3p+3}{2}.
\]

There are

\[
\binom82=28
\]

such pairs.

### 4.2 Nearest--adjacent pairs

The relative mask of \(N_{a,u}\) and any \(A_b\) is balanced.

- If \(a=b\), phases agree and the mutual distance is \(4p\).
- If \(a\ne b\), phases disagree everywhere and (3.3) again gives \(4p\).

Therefore every nearest--adjacent pair has

\[
|D_{N_{a,u}}\cap D_{A_b}|
=
\frac{d_N+d_A-4p}{2}
=
\frac{3p+3}{2}.
\]

There are

\[
56\cdot8=448
\]

such pairs.

### 4.3 Nearest--nearest pairs with different characters

For \(u\ne v\), the relative mask

\[
m_{a,u}m_{b,v}
=
(-1)^{(u+v)\cdot x+u\cdot a+v\cdot b}
\]

is a nonconstant character and is therefore balanced. Its mutual distance is
\(4p\), both when \(a=b\) and when \(a\ne b\). Equation (4.1) gives

\[
|D_{N_{a,u}}\cap D_{N_{b,v}}|
=
d_N-2p
=
\frac{3p-1}{2}.
\]

The count is

\[
8\binom72+\binom82(7\cdot6)
=
168+1176
=
1344.
\]

The first term has a common aligned layer; the second chooses two distinct
aligned layers and two distinct nonzero characters.

### 4.4 Nearest--nearest pairs with the same character

If \(u=v\), distinct states require \(a\ne b\), and the relative mask is the
constant

\[
(-1)^{u\cdot(a+b)}.
\]

For fixed distinct \(a,b\), the nonzero vector \(a+b\) defines a nonzero linear
functional on \(V\). Among the seven nonzero \(u\)'s:

- three satisfy \(u\cdot(a+b)=0\);
- four satisfy \(u\cdot(a+b)=1\).

In the first case the relative mask is all positive, the mutual distance is
\(4p+4\), and

\[
|D_i\cap D_j|=d_N-(2p+2)=\frac{3p-5}{2}.
\]

Its multiplicity is

\[
\binom82\cdot3=84.
\]

In the second case the relative mask is all negative, the mutual distance is
\(4p-4\), and

\[
|D_i\cap D_j|=d_N-(2p-2)=\frac{3p+3}{2}.
\]

Its multiplicity is

\[
\binom82\cdot4=112.
\]

### 4.5 Complete orbit certificate

The five pair classes are therefore

\[
\begin{array}{c|c|c}
\text{pair class}&|D_i\cap D_j|&\text{multiplicity}\\ \hline
NN,\ u=v,\ u\cdot(a+b)=0&(3p-5)/2&84\\
NN,\ u\ne v&(3p-1)/2&1344\\
NN,\ u=v,\ u\cdot(a+b)=1&(3p+3)/2&112\\
NA&(3p+3)/2&448\\
AA&(3p+3)/2&28.
\end{array}
\tag{4.2}
\]

Their multiplicities sum to

\[
84+1344+112+448+28=2016=\binom{64}{2}.
\]

Thus (4.2) covers every unordered pair exactly once; it is not an extrapolation
from small-prime enumeration.

## 5. The omitted adjacent shell has nonzero leading weight

Define

\[
\beta_q(n)
=
\Pr\!\left[\operatorname{Bin}(n,q)\ge\left\lceil n/2\right\rceil\right].
\]

Because \(p\equiv3\pmod4\), \(d_N\) and \(d_N+4\) are even. For even \(n\), let

\[
t_n={n\choose n/2}[q(1-q)]^{n/2}
\]

be the central mass. Successive upper-tail masses have limiting ratio
\(\rho=q/(1-q)<1\), and every such ratio is at most \(\rho\). Dominated
convergence therefore gives

\[
\frac{\beta_q(n)}{t_n}
\longrightarrow
\frac1{1-\rho}
=
\frac{1-q}{1-2q}.
\tag{5.1}
\]

The common factor in (5.1) cancels between \(n\) and \(n+4\). Moreover,

\[
\frac{t_{n+4}}{t_n}
=
\frac{(n+4)(n+3)(n+2)(n+1)}
{(n/2+2)^2(n/2+1)^2}
[q(1-q)]^2
\longrightarrow
[4q(1-q)]^2.
\tag{5.2}
\]

Consequently,

\[
\frac{8\beta_q(d_N+4)}{56\beta_q(d_N)}
\longrightarrow
\frac{[4q(1-q)]^2}{7}>0.
\tag{5.3}
\]

Thus the eight \(d_N+4\) states cannot be placed in a little-\(o\) remainder.
Equation (5.3) already falsifies the old auxiliary farther-shell condition.
The event-union normalization itself is falsified after the overlap argument
below rules out asymptotic absorption of those events by the nearest-event
union.

## 6. Every other shell is negligible

Let

\[
I_q=D(1/2\|q)=-\frac12\log(4q(1-q))>0.
\]

For an even distance \(m\), the Chernoff bound and the central mass lower bound
give

\[
\beta_q(m)\le e^{-mI_q},
\qquad
\beta_q(d_N)\ge c\,d_N^{-1/2}e^{-d_NI_q}
\tag{6.1}
\]

for a fixed positive constant \(c\) and sufficiently large \(p\). The closest
nonleading shell is

\[
4p-4=d_N+\frac{p-7}{2},
\]

and there are fewer than \(16p\) nonleading states. If \(R_p(q)\) is the sum of
their pairwise event probabilities, then (6.1) yields

\[
\frac{R_p(q)}{56\beta_q(d_N)}
\le
C p^{3/2}
\exp\!\left[-\frac{p-7}{2}I_q\right]
\longrightarrow0.
\tag{6.2}
\]

The same conclusion holds after normalizing by the larger repaired leading
term.

## 7. Leading pair intersections are negligible

Let \(i,j\) be any of the 2,016 leading pairs. Partition their disagreement
sets into

\[
C=D_i\cap D_j,\qquad
A=D_i\setminus D_j,\qquad
B=D_j\setminus D_i.
\]

By (4.2), uniformly over the five fixed pair classes,

\[
\frac{|C|}{p}\to\frac32,\qquad
\frac{|A|}{p}\to2,\qquad
\frac{|B|}{p}\to2.
\tag{7.1}
\]

Let \(x,y,z\) be the empirical flip fractions in \(C,A,B\). Cramér's theorem
gives the three-region rate

\[
F_q(x,y,z)
=
\frac32D(x\|q)+2D(y\|q)+2D(z\|q).
\tag{7.2}
\]

The limiting constraints for the two events are

\[
\frac32x+2y\ge\frac74,
\qquad
\frac32x+2z\ge\frac74.
\tag{7.3}
\]

For the first constraint alone, strict convexity and Jensen's inequality give
the unique minimizer

\[
(x,y,z)=\left(\frac12,\frac12,q\right)
\]

and rate

\[
I_q^{\mathrm{single}}
=
\frac72D(1/2\|q).
\tag{7.4}
\]

At this minimizer, the left side of the second constraint is

\[
\frac34+2q<\frac74
\]

for every fixed \(q<1/2\). The joint feasible set in (7.3) is compact, and the
lower-semicontinuous rate (7.2) has no point at the single-event minimum.
Therefore its joint minimum satisfies

\[
J_q>I_q^{\mathrm{single}}.
\tag{7.5}
\]

The \(O(1)\) differences among the five exact classes and the ceiling in each
event threshold do not alter these rates. Hence

\[
\Pr(E_i^{\ge}\cap E_j^{\ge})
=
\exp[-pJ_q+o(p)]
=
o\!\left(\exp[-pI_q^{\mathrm{single}}+o(p)]\right)
=
o(B_p(q)),
\tag{7.6}
\]

where

\[
B_p(q)=56\beta_q(d_N)+8\beta_q(d_N+4).
\]

Because the number of leading pairs is the fixed number 2,016, summing (7.6)
preserves the little-\(o\) relation.

## 8. Bonferroni theorem and transmitted-state scope

Let \(U_{L,p}^{\ge}\) be the union over the 64 leading events, and let \(S_{2,p}\)
be the sum of all their pair intersections. First-order Bonferroni and the
ordinary union bound give

\[
B_p(q)-S_{2,p}
\le
\Pr(U_{L,p}^{\ge})
\le
\Pr(U_p^{\ge})
\le
B_p(q)+R_p(q).
\tag{8.1}
\]

Sections 6 and 7 give \(R_p(q)=o(B_p(q))\) and
\(S_{2,p}=o(B_p(q))\). Dividing (8.1) by \(B_p(q)\) proves

\[
\boxed{
\Pr(U_p^{\ge})
=
\left[
56\beta_q\!\left(\frac{7p-1}{2}\right)
+8\beta_q\!\left(\frac{7p+7}{2}\right)
\right](1+o(1)).
}
\tag{8.2}
\]

Combining (8.2) with (5.3) gives

\[
\frac{\Pr(U_p^{\ge})}{56\beta_q(d_N)}
\longrightarrow
1+\frac{[4q(1-q)]^2}{7},
\tag{8.3}
\]

not one.

The derivation above used the canonical truth only to simplify notation. For an
arbitrary transmitted shift and mask of a fixed type, apply the following
Hamming isometry independently in each layer:

1. rotate coordinates to remove the transmitted phase;
2. multiply by the transmitted mask sign.

The BSC is invariant under this coordinate permutation and sign change.
Closure of \(\mathcal M\) under multiplication permutes the candidate masks.
For a type-0 truth, the wrong-type relative signature remains \(x+s\); for a
type-1 truth it becomes \(-x+s\). In either case the eight layer residues are
distinct, one layer aligns at each of eight shifts, and the phase difference
between any two candidates is constant across layers. Sections 3 and 4
therefore apply unchanged. Thus (8.2) holds for every transmitted state in this
exact frozen two-type bank.

Equivalently, the finite bank admits three explicit isometric actions. With
\(h=7/2\) in \(\mathbb Z_p\), they act on state labels as

\[
\begin{aligned}
A_r(t,s,m)&=(t,s-r,m),\\
B_n(t,s,m)&=(t,s,mn),\\
C(t,s,m)&=(1-t,\ s+(2t-1)h,\ m\circ\iota),
\end{aligned}
\tag{8.4}
\]

where \(r\in\mathbb Z_p\), \(n\in\mathcal M\), and
\(\iota(x)=7-x\) reverses the eight layers. \(A_r\) is a common cyclic
coordinate permutation, \(B_n\) is layerwise sign multiplication, and \(C\)
is the layer-reversal/type-swap coordinate permutation. Direct substitution
in \(h_{t,s,m}\) gives (8.4), \(C^2=1\), and closure because
\(\mathcal M\) is closed under multiplication and layer reversal. The
\(2\cdot p\cdot16=32p\) normal forms \(C^\epsilon A_rB_n\) send the canonical
state to all 32\(p\) bank states exactly once. Hamming distances,
disagreement intersections, and i.i.d. BSC probabilities are invariant under
these actions.

This last reduction is special to the two signatures \(0\) and \(x\), the
character mask group, and the i.i.d. BSC. It is not a statement about arbitrary
overlap-one signatures.

## 9. Canonical-first decoder corollaries and balanced closure

### 9.1 Strict leading tail

Every leading distance is even. A strict-win event uses

\[
\beta_q^{>}(d)
=
\Pr[\operatorname{Bin}(d,q)\ge d/2+1]
\]

instead of the inclusive tail. Central-term domination gives

\[
\frac{\beta_q^{>}(d)}{\beta_q(d)}
\longrightarrow
\frac{q}{1-q},
\tag{9.1}
\]

which is nonzero and is not one. A deterministic truth-favoring,
wrong-state-favoring, canonical-index, or randomized tie rule can therefore
change the leading constant.

For canonical type-zero truth, an exact-Hamming decoder using the frozen
lowest-type-ID rule chooses type zero on an exact cross-type score tie. A
wrong-type state must therefore strictly beat the truth. Define

\[
S_p(q)
=
56\beta_q^{>}(d_N)+8\beta_q^{>}(d_N+4).
\tag{9.2}
\]

The strict leading-pair intersections are subsets of their inclusive
counterparts. Since (9.1) makes \(S_p(q)\) a positive fixed-\(q\) fraction of
\(B_p(q)\), Sections 6 and 7 also give a strict wrong-type event-union
asymptotic

\[
\Pr\!\left(\bigcup_{i\text{ wrong type}}E_i^{>}\right)
=
S_p(q)(1+o(1)).
\tag{9.3}
\]

### 9.2 Correct-type interference

For canonical truth, the nontransmitted correct-type states have the exact
spectrum

\[
\begin{array}{c|c|l}
\text{distance}&\text{multiplicity}&\text{source}\\ \hline
4p-4&p-1&\text{nonzero shift, all negative}\\
4p&14p&\text{balanced masks at all shifts}\\
4p+4&p-1&\text{nonzero shift, all positive}\\
8p&1&\text{zero shift, all negative}.
\end{array}
\tag{9.4}
\]

To derive (9.4), apply (3.3) to each of the \(p-1\) nonzero shifts. At zero
shift, the fourteen balanced masks differ from the truth in four complete
layers and contribute fourteen additional states at \(4p\); the all-negative
mask contributes the state at \(8p\); the all-positive mask is the transmitted
state and is excluded. The multiplicities sum to \(16p-1\).

The correct-type minimum \(4p-4\) is separated from \(d_N\) by
\((p-7)/2\). The same estimate as (6.2) makes the probability that any
nontransmitted correct-type state ties or beats the truth \(o(S_p(q))\).

Let \(G_p\) be the strict wrong-type event union in (9.3), let \(C_p\) be the
correct-type interference union, and let \(F_p\) be the actual type-decoder
error event. Because the decoder favors type zero on an exact cross-type tie,

\[
\Pr(G_p)-\Pr(C_p)
\le
\Pr(F_p)
\le
\Pr(G_p).
\]

Therefore

\[
\boxed{
\Pr(\text{decoded type}\ne0\mid h_{0,0,\mathbf1})
=
S_p(q)(1+o(1)).
}
\tag{9.5}
\]

This proves `PRW-T1D-CANONICAL-FIRST-TYPE0` in the frozen canonical state. The
actions \(A_rB_n\) from (8.4) preserve the type label, preserve the
lowest-type-ID tie policy, and act transitively on the \(16p\) states of each
fixed type. Equation (9.5) therefore holds for every transmitted type-zero
state. In particular,

\[
\frac{S_p(q)}{B_p(q)}
\longrightarrow
\frac{q}{1-q};
\]

at \(q=0.20\), the canonical-first decoder retains asymptotically one quarter
of the inclusive event-union leading constant.

### 9.3 Type-one inclusive-tie corollary

For type-one truth, the production helper's same lowest-type-ID rule favors
wrong type zero on an exact cross-type score tie. The relevant wrong-type event
is therefore the inclusive union from (8.2), not the strict union from (9.3).
Let \(H_p\) be that inclusive wrong-type event union, let \(C_p\) retain the
correct-type interference meaning from Section 9.2, and let \(F_{1,p}\) be
actual class error conditional on type-one truth. Then

\[
\Pr(H_p)-\Pr(C_p)
\le
\Pr(F_{1,p})
\le
\Pr(H_p).
\tag{9.6}
\]

The lower bound holds because, when no correct-type competitor ties or beats
the truth, any wrong type-zero state that ties or beats it is selected. The
upper bound holds because a type error requires some wrong-type state to tie or
beat the transmitted state. Sections 8 and 9.2 give

\[
\Pr(H_p)=B_p(q)(1+o(1)),
\qquad
\Pr(C_p)=o(B_p(q)).
\]

Consequently,

\[
\boxed{
\Pr(\text{decoded type}\ne1\mid T=1)
=
B_p(q)(1+o(1)).
}
\tag{9.7}
\]

Again \(A_rB_n\) preserves the type label and is transitive within the
type-one half of the bank, so (9.7) holds for every transmitted type-one state.
This proves `PRW-T1D-CANONICAL-FIRST-TYPE1` in the frozen bank.

### 9.4 Frozen 50/50 balanced-type theorem

Freeze the transmitted-type distribution before evaluation:

\[
\Pr(T=0)=\Pr(T=1)=\frac12.
\tag{9.8}
\]

The conditional distribution within either type may be any frozen
distribution because the corresponding error asymptotic is identical for
every state of that type. Averaging (9.5) and (9.7) yields

\[
\boxed{
\Pr(\text{decoded type}\ne T)
=
\frac{S_p(q)+B_p(q)}{2}(1+o(1)).
}
\tag{9.9}
\]

Relative to the inclusive leading term,

\[
\frac{S_p(q)+B_p(q)}{2B_p(q)}
\longrightarrow
\frac12\left(\frac{q}{1-q}+1\right)
=
\frac{1}{2(1-q)}.
\tag{9.10}
\]

At \(q=0.20\), the frozen balanced exact-Hamming decoder therefore has leading
ratio \(5/8=0.625\). This closes the analytic
`PRW-T1D-ALL-STATES` corollary only for the stated lowest-type-ID mathematical
tie rule, the frozen two-type bank, fixed
\(q\in(0,1/2)\), and total transmitted-type mass \(1/2\) per type. It is not a
claim for an unfrozen empirical type prior, a different tie rule, additional
types, another bank, or a numerical scorer that does not preserve mathematical
ties.

The 2026-07-26 production-path audit found exactly that last mismatch. For both
truth types, every one of the 56 leading midpoint observations, and
`p in {11,19,31}`, integer Hamming and direct dot-product scoring returned
`336/336` cross-type ties. The current float-FFT dense scorer retained only
`117/336` as bit-exact ties; margins up to
`3.3306690738754696e-16` sometimes changed the winner. This does not alter the
event-union proof. It invalidates binding its strict/inclusive constants to the
current float-FFT implementation without a declared numerical-tie contract.

## 10. Near-half nonuniformity

The theorem is pointwise in fixed \(q<1/2\). It is not uniform as
\(q\uparrow1/2\). If \(q=1/2-\varepsilon\), then

\[
I_q=-\frac12\log(1-4\varepsilon^2)
=
2\varepsilon^2+O(\varepsilon^4).
\]

The nonleading suppression in (6.2) competes with an \(O(p)\) multiplicity.
The relevant scale is therefore \(p\varepsilon^2\) versus \(\log p\), not \(p\)
alone. A sequence \(q_p\to1/2\) requires a separate theorem.

Stable finite computations illustrate the boundary:

\[
\begin{array}{c|c|c}
p&q&R_p(q)/B_p(q)\\ \hline
4691&0.20&5.55\times10^{-225}\\
4691&0.45&8.34\times10^{-3}\\
4691&0.49&6.93\times10^{2}\\
100003&0.49&1.06.
\end{array}
\]

At \(p=31\), the leading pair-sum ratio \(S_{2,p}/B_p\) is approximately
`0.0004256` at \(q=0.20\), but `11.57` at \(q=0.45\) and `19.18` at
\(q=0.49\). Ratios above one do not contradict Bonferroni or the fixed-\(q\)
limit; they mean the finite first-order lower bound is uninformative there.

No `p=4691` result should be described as a finite near-half certificate
without reporting these remainder and intersection terms.

## 11. Executable cross-checks

The proof is algebraic; enumeration is a falsifier and regression check rather
than its logical basis. The current implementation cross-checks:

- the seven-shell formulas against materialized banks at \(p=11,19,31\);
- the five overlap classes against all 2,016 materialized leading pairs at
  those primes;
- the closed-form shell counts at \(p=4691\);
- stable binomial-tail ratios at \(p=4691\);
- exact finite pair-sum ratios of approximately `0.2906144`, `0.02026685`,
  `0.000425626`, and `9.57689e-6` at \(p=11,19,31,43\), respectively, for
  \(q=0.20\);
- the abstract lowest-type-ID tie helper's exact outcome for both transmitted
  types;
- the type-one inclusive-to-inclusive ratio of one; and
- the exact finite balanced identity
  \(\tfrac12(1+S_p(q)/B_p(q))\), converging to \(1/[2(1-q)]\).

An additional read-only audit enumerated every possible transmitted state at
\(p=11,19,31\); every state had the same wrong-type spectrum and the same five
leading orbit counts. This supports, but is not needed in place of, the
isometry argument in Section 8.

A separate production-path falsifier constructed all 336 leading midpoint
cases described in Section 9. Direct scoring preserved every mathematical tie;
the float-FFT scorer did not. The current tests therefore cross-check the
analytic formulas but do not validate finite production-decoder error.

## 12. Publication-readiness verdict

The seven shell multiplicities, five overlap multiplicities, tail-ratio
correction, nonleading bound, joint large-deviation separation, and
Bonferroni squeeze are derived in the exact stated model. No combinatorial
lemma used by `PRW-T1R-TE-EVENT-UNION` remains open.

The asymmetric tie corollary is closed only for an exact-Hamming decoder over
every state of both types and the frozen 50/50 type mixture. The current
float-FFT production linkage is failed/open. This is an internal analytic
method-development result, not production or application evidence.

The broader research claim is **not publication-ready**:

1. The bounded
   [primary-source claim chart](prime-ring-t1r-primary-claim-chart-2026-07.md)
   found a direct 1992 collision for the Legendre-inner/outer-code construction
   and unmasked distance identity, but no source in its declared search set
   stating the exact event-union theorem. That is bounded positioning, not
   novelty evidence; independent proof-equivalence review and broader
   citation/index coverage remain required.
2. No extension is proved for arbitrary signatures, arbitrary mask families,
   \(q_p\to1/2\), growing layer count, or non-i.i.d. perturbations.
3. The result supplies no demonstrated retrieval, training, latency, memory,
   or cost advantage.

The defensible current description is:

> a proved, narrow asymptotic correction for the tie-as-error wrong-type event
> union of one frozen Legendre-by-\(RM(1,3)\) bank, with strict type-zero,
> inclusive type-one, and frozen 50/50 balanced exact-Hamming tie semantics;
> the current float-FFT implementation does not preserve all mathematical ties,
> and novelty and systems utility remain unresolved.
