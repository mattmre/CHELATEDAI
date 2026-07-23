"""Dictionary-misspecification sensitivity.

The Tier-0 wedge assumes the decoder knows A EXACTLY. In any real frozen
embedding space A must be estimated, and sparse dictionary learning is provably
non-identifiable (arXiv:2512.05534). This sweep asks: how fast does the l1
advantage die as the decoder's dictionary drifts from the true one?

A_hat = normalize(A + eps * G),  eps in {0, .05, .1, .2, .4}
The LINEAR probe is trained on true (x, s) pairs and is therefore UNAFFECTED by
eps -- it is the fixed reference. Only the nonlinear arm degrades.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, r"C:\Users\mattm\AppData\Local\Temp\claude\D--GITHUB-CHELATEDAI--claude-worktrees-relaxed-wozniak-271e04\00b9ffaa-9115-4aad-9219-bcc10f287c90\scratchpad")
from wedge_pilot import make_dict, sample_S, fista, metrics  # noqa

d, m, k = 128, 4096, 8
n_train, n_test = 20000, 1000
rng = np.random.default_rng(1)
A = make_dict(d, m, rng)

Str = sample_S(m, k, n_train, rng); Xtr = A @ Str
XXt = Xtr @ Xtr.T; XSt = Xtr @ Str.T
Sv = sample_S(m, k, 1000, rng); Xv = A @ Sv
best, W = -1.0, None
for lam in [1e-6, 1e-4, 1e-2, 1.0, 1e2]:
    Wc = np.linalg.solve(XXt + lam * np.eye(d), XSt).T
    f, _ = metrics(Wc @ Xv, Sv, k)
    if f.mean() > best:
        best, W = f.mean(), Wc

Ste = sample_S(m, k, n_test, rng); Xte = A @ Ste
f_lin, e_lin = metrics(W @ Xte, Ste, k)
print(json.dumps(dict(arm="linear", frac=float(f_lin.mean()), exact=float(e_lin.mean()))), flush=True)

for eps in [0.0, 0.05, 0.1, 0.2, 0.4]:
    t0 = time.time()
    Ah = A + eps * rng.standard_normal(A.shape) / np.sqrt(d)
    Ah /= np.linalg.norm(Ah, axis=0, keepdims=True)
    coh = float(np.abs((Ah * A).sum(axis=0)).mean())  # mean column alignment
    Sl1 = fista(Ah, Xte, 0.02 * np.abs(Ah.T @ Xte).max(), iters=250)
    f_l1, e_l1 = metrics(Sl1, Ste, k)
    df = f_l1 - f_lin
    de = e_l1 - e_lin
    print(json.dumps(dict(eps=eps, col_align=round(coh, 4),
                          l1_frac=float(f_l1.mean()), l1_exact=float(e_l1.mean()),
                          frac_diff_mean=float(df.mean()), frac_diff_sd=float(df.std(ddof=1)),
                          exact_diff_mean=float(de.mean()), exact_diff_sd=float(de.std(ddof=1)),
                          secs=round(time.time() - t0, 1))), flush=True)
