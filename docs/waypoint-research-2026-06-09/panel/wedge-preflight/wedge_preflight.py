"""Preflight for the within-space wedge test (Route A instrument).

Question: is there a (d, k, m) window where the POPULATION-OPTIMAL LINEAR decoder
of a k-sparse superposed code fails while an l1 (compressed-sensing) decoder with
the TRUE dictionary succeeds?

This is a design-calibration run, NOT the experiment. Everything uses the true
dictionary F, so the l1 arm is an ORACLE arm.
"""
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def make_dict(d, m, g):
    F = torch.randn(d, m, generator=g, device=DEV)
    F = F / F.norm(dim=0, keepdim=True)
    return F


def sample(F, k, n, g, sigma=0.01):
    d, m = F.shape
    # exactly-k support per row, uniform without replacement
    scores = torch.rand(n, m, generator=g, device=DEV)
    idx = scores.topk(k, dim=1).indices
    a = torch.zeros(n, m, device=DEV)
    vals = 0.5 + torch.rand(n, k, generator=g, device=DEV)  # U[0.5,1.5], positive
    a.scatter_(1, idx, vals)
    x = a @ F.T
    x = x + sigma * torch.randn_like(x)
    return x, a


def linear_decoder(F, k, n_train, g, sigma):
    """Population-optimal (MMSE) linear decoder a_hat = W x, fit with free data."""
    x, a = sample(F, k, n_train, g, sigma)
    Sxx = (x.T @ x) / n_train
    Sax = (a.T @ x) / n_train
    d = F.shape[0]
    W = torch.linalg.solve(Sxx + 1e-6 * torch.eye(d, device=DEV), Sax.T).T
    return W


def fista(F, x, lam, iters=300, nonneg=True):
    """min_a 0.5||x - F a||^2 + lam ||a||_1, batched over rows of x."""
    n = x.shape[0]
    m = F.shape[1]
    L = torch.linalg.matrix_norm(F, ord=2) ** 2
    step = 1.0 / L
    a = torch.zeros(n, m, device=DEV)
    y = a.clone()
    t = 1.0
    for _ in range(iters):
        grad = (y @ F.T - x) @ F
        z = y - step * grad
        a_new = torch.sign(z) * torch.clamp(z.abs() - step * lam, min=0)
        if nonneg:
            a_new = torch.clamp(a_new, min=0)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        y = a_new + ((t - 1) / t_new) * (a_new - a)
        a, t = a_new, t_new
    return a


def r2(a_hat, a):
    ss_res = ((a_hat - a) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def auc_support(a_hat, a):
    """AUC for detecting active coordinates, computed by rank statistic."""
    s = a_hat.flatten()
    lab = (a.flatten() > 0).float()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(len(s), device=DEV, dtype=s.dtype) + 1
    npos = lab.sum()
    nneg = len(s) - npos
    return (((ranks * lab).sum() - npos * (npos + 1) / 2) / (npos * nneg)).item()


def run_cell(d, m, k, sigma=0.01, n_train=20000, n_test=4000, n_val=2000):
    g = torch.Generator(device=DEV).manual_seed(1234 + d * 7 + k)
    F = make_dict(d, m, g)
    W = linear_decoder(F, k, n_train, g, sigma)

    xv, av = sample(F, k, n_val, g, sigma)
    best = (-1e9, None)
    for lam in [0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4]:
        ah = fista(F, xv, lam)
        sc = r2(ah, av)
        if sc > best[0]:
            best = (sc, lam)
    lam = best[1]

    xt, at = sample(F, k, n_test, g, sigma)
    lin = xt @ W.T
    cs = fista(F, xt, lam)
    return dict(d=d, m=m, k=k, lam=lam,
                lin_r2=r2(lin, at), cs_r2=r2(cs, at),
                lin_auc=auc_support(lin, at), cs_auc=auc_support(cs, at))


if __name__ == "__main__":
    m = 4096
    rows = []
    for k in [4, 8, 16, 32]:
        for d in [32, 64, 128, 256, 512]:
            r = run_cell(d, m, k)
            rows.append(r)
            print(f"k={k:3d} d={d:4d} m={m}  lam={r['lam']:.3f}  "
                  f"linR2={r['lin_r2']:+.3f} csR2={r['cs_r2']:+.3f}  "
                  f"linAUC={r['lin_auc']:.3f} csAUC={r['cs_auc']:.3f}  "
                  f"dAUC={r['cs_auc']-r['lin_auc']:+.3f}", flush=True)
