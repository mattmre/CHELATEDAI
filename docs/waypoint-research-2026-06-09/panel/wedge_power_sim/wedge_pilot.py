"""Pilot power simulation for the within-space superposition wedge test.

Question: in ONE frozen space with KNOWN superposed sparse features, does a
gradient-free nonlinear readout (l1 / basis pursuit, FISTA) beat the strongest
LINEAR probe (full-fit ridge trained on 20k labelled samples) at recovering
which features are active?

Both methods get full information (linear probe is trained on ground-truth
labels; l1 gets the dictionary A). Any gap is therefore FUNDAMENTAL
(interference / span-exhaustion), not informational.
"""
import numpy as np
import time
import json
import sys

rng_global = np.random.default_rng(0)


def make_dict(d, m, rng):
    A = rng.standard_normal((d, m)) / np.sqrt(d)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def sample_S(m, k, n, rng):
    S = np.zeros((m, n))
    idx = np.argsort(rng.random((n, m)), axis=1)[:, :k]  # k distinct per column
    vals = rng.uniform(0.5, 1.5, size=(n, k)) * rng.choice([-1.0, 1.0], size=(n, k))
    rows = idx.ravel()
    cols = np.repeat(np.arange(n), k)
    S[rows, cols] = vals.ravel()
    return S


def fista(A, X, lam, iters=200):
    """Vectorised FISTA for 0.5||X - A S||^2 + lam ||S||_1 over columns of X."""
    m = A.shape[1]
    n = X.shape[1]
    L = np.linalg.norm(A, 2) ** 2
    S = np.zeros((m, n))
    Z = S.copy()
    t = 1.0
    At = A.T
    for _ in range(iters):
        G = At @ (A @ Z - X)
        Snew = Z - G / L
        Snew = np.sign(Snew) * np.maximum(np.abs(Snew) - lam / L, 0.0)
        tnew = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = Snew + ((t - 1) / tnew) * (Snew - S)
        S, t = Snew, tnew
    return S


def topk_support(Shat, k):
    return np.argpartition(-np.abs(Shat), k, axis=0)[:k, :]


def metrics(Shat, S, k):
    n = S.shape[1]
    true_sup = np.argpartition(-np.abs(S), k, axis=0)[:k, :]
    pred_sup = topk_support(Shat, k)
    frac = np.empty(n)
    for j in range(n):
        frac[j] = len(np.intersect1d(true_sup[:, j], pred_sup[:, j])) / k
    exact = (frac == 1.0).astype(float)
    return frac, exact


def run_cell(d, m, k, sigma=0.0, n_train=20000, n_test=1000, seed=0, lam_scale=0.02):
    rng = np.random.default_rng(seed)
    A = make_dict(d, m, rng)
    # --- train the linear probe (strongest fair baseline: full-fit ridge x -> s)
    Str = sample_S(m, k, n_train, rng)
    Xtr = A @ Str
    if sigma:
        Xtr += sigma * rng.standard_normal(Xtr.shape)
    XXt = Xtr @ Xtr.T
    XSt = Xtr @ Str.T
    # pick ridge lambda by held-out (validation) support recovery -- baseline
    # gets its own hyperparameter tuned, same as the treatment.
    Sv = sample_S(m, k, 1000, rng)
    Xv = A @ Sv
    if sigma:
        Xv += sigma * rng.standard_normal(Xv.shape)
    bestscore, bestW = -1.0, None
    for lam in [1e-6, 1e-4, 1e-2, 1.0, 1e2]:
        W = np.linalg.solve(XXt + lam * np.eye(d), XSt).T
        f, _ = metrics(W @ Xv, Sv, k)
        if f.mean() > bestscore:
            bestscore, bestW = f.mean(), W
    W = bestW

    Ste = sample_S(m, k, n_test, rng)
    Xte = A @ Ste
    if sigma:
        Xte += sigma * rng.standard_normal(Xte.shape)

    f_lin, e_lin = metrics(W @ Xte, Ste, k)
    lam_l1 = lam_scale * np.abs(A.T @ Xte).max()
    Sl1 = fista(A, Xte, lam_l1, iters=250)
    f_l1, e_l1 = metrics(Sl1, Ste, k)

    return dict(d=d, m=m, k=k, sigma=sigma,
                lin_frac=float(f_lin.mean()), l1_frac=float(f_l1.mean()),
                lin_exact=float(e_lin.mean()), l1_exact=float(e_l1.mean()),
                d_frac=f_l1 - f_lin, d_exact=e_l1 - e_lin,
                ridge_val=float(bestscore))


CELLS = [
    (128, 4096, 8, 0.0),
    (128, 4096, 8, 0.05),
    (96, 4096, 6, 0.0),
    (256, 8192, 12, 0.0),
    (64, 2048, 5, 0.0),
    (512, 4096, 8, 0.0),   # linear-sufficient control: d > k^2 log m
]

out = []
for (d, m, k, sg) in (CELLS if __name__ == "__main__" else []):
    t0 = time.time()
    r = run_cell(d, m, k, sigma=sg, n_test=1000, seed=1)
    df, de = r.pop("d_frac"), r.pop("d_exact")
    r["frac_diff_mean"] = float(df.mean())
    r["frac_diff_sd"] = float(df.std(ddof=1))
    r["exact_diff_mean"] = float(de.mean())
    r["exact_diff_sd"] = float(de.std(ddof=1))
    r["k_log_m_over_k"] = float(k * np.log(m / k))
    r["k2_log_m"] = float(k * k * np.log(m))
    r["secs"] = round(time.time() - t0, 1)
    out.append(r)
    print(json.dumps(r), flush=True)

if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else "pilot.json", "w") as fh:
        json.dump(out, fh, indent=2)
