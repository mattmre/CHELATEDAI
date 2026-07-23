# 3. System Under Test

We evaluate a post-hoc correction layer placed between an embedding encoder and cosine retrieval. Let
$x\in\mathbb{R}^d$ be an encoder output and $n(x)=x/\lVert x\rVert_2$ (with the implementation's
usual nonzero-vector assumption). Every adapter returns an L2-normalized vector, so retrieval ranks
documents by the cosine-equivalent dot product between normalized query and document vectors.

## 3.1 Adapter forward maps

The residual MLP used by the supervised loop is

$$g_\theta(x)=n\!\left(x+W_2\,\mathrm{ReLU}(W_1x+b_1)+b_2\right).$$

Its hidden width defaults to $d/2$. Both linear layers are initialized with zero bias and Gaussian
weights of standard deviation $10^{-3}$, making the initial map close to identity. The nonlinear
comparison in §5.5 uses the same residual form with a 1,024-wide GELU hidden layer and no bounded
wrapper.

The D1 ladder baseline is **closed-form orthogonal Procrustes**: it computes an orthogonal map by
SVD of the paired-document cross-covariance. It is not the trainable Cayley adapter described next,
and the two objects must not share an unqualified "Procrustes" label in the ladder or its caption.

The trainable Cayley adapter parameterizes $A=P-P^\top$ and computes the map

$$W=(I-A)(I+A)^{-1},\qquad g_{\mathrm{proc}}(x)=n\!\left((xW^\top)\odot s\right),$$

where $s$ is a learned per-dimension scale initialized to one. The low-rank affine alternative is

$$g_{\mathrm{lr}}(x)=n\!\left(x+(xU)V^\top+b\right),$$

with $U\in\mathbb{R}^{d\times r}$, $V\in\mathbb{R}^{d\times r}$, and $V$ initialized to zero,
so its initial correction is exactly zero.

## 3.2 Exact bounded-adapter map and correction floor

The bounded wrapper operates on the normalized input $\bar x=n(x)$ and the already normalized output
$z=g_\theta(x)$. With learned dimension scale $s$ (initialized to all ones), define

$$\delta=((z-\bar x)\odot s),\qquad r=\lVert\delta\rVert_2.$$

For lower and upper correction bounds $a$ and $b$, the implementation applies

$$
\gamma(r)=
\begin{cases}
a/r, & 10^{-10}<r<a,\\
b/r, & r>b,\\
1, & \text{otherwise},
\end{cases}
\qquad
g_{[a,b]}(x)=n\!\left(\bar x+\gamma(r)\delta\right).
$$

Thus the lower bound is a floor only for a *nonzero* correction: an exactly zero (or numerically
$\le10^{-10}$) correction is left at zero. C3a uses $a=0.01$ and
$b=0.5$. The stated INT8 scale is $1/128=0.0078125$,
so $a=0.01$ is intended to place nonzero corrections above that quantization-noise scale. After
clipping, the vector is normalized again. The wrapper therefore bounds the pre-renormalization
correction norm; it does not imply an identical Euclidean displacement after the final normalization.

## 3.3 Drift-triggered controller

The controller stores a scalar temperature $T$. Given a nonnegative drift magnitude $m$, observation
is a strict-threshold update:

$$
T\leftarrow
\begin{cases}
\max(T,\min(T_{\max},m)), & m>\tau,\\
T, & m\le\tau.
\end{cases}
$$

Correction is enabled iff $T>1e-06$. With
$\rho=\mathrm{clip}(T/T_{\max},0,1)$, a correction cycle receives learning-rate scale
$0.1+0.9\rho$, epoch count $1+\mathrm{round}(2\rho)$, and online intensity $\rho$. At cycle end,
$T\leftarrow cT$ and values at or below $1e-06$ are set to zero. Constructor
defaults are $T_0=0.0$, $\tau=0.15$,
$T_{\max}=1.0$, and $c=0.7$.

For the supervised query-encoder-swap cycle, the actual drift signal is
$max(0, baseline_ndcg - current_ndcg)$, the campaign sets $\tau=0$, $T_{\max}=1$, and uses
$c=0.5$. When the trigger fires and anchor pairs exist, the adapter is
reinitialized from the run seed, trained by query–document InfoNCE, applied to every cached document,
and written back. In the default non-compounding regime every cycle applies the newly trained adapter
to the same pre-correction document snapshot, making the correction a reproducible one-shot map rather
than an accumulated trajectory.
