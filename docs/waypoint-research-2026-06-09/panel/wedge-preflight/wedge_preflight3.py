"""Preflight 3: (a) real bootstrap half-widths for the Route A decision statistic,
(b) the Route B 'linear cannot do modular structure' claim, tested literally."""
import torch
import numpy as np
from wedge_preflight import make_dict, sample, fista, DEV
from wedge_preflight2 import linear_detector, per_feature_auc

torch.manual_seed(0)


def per_feature_auc_vec(scores, a):
    lab = (a > 0).float()
    n, m = scores.shape
    order = torch.argsort(scores, dim=0)
    ranks = torch.empty_like(scores)
    ar = (torch.arange(n, device=DEV, dtype=scores.dtype) + 1).unsqueeze(1).expand(n, m)
    ranks.scatter_(0, order, ar)
    npos = lab.sum(0)
    nneg = n - npos
    auc = ((ranks * lab).sum(0) - npos * (npos + 1) / 2) / (npos * nneg).clamp(min=1)
    return auc, (npos > 0) & (nneg > 0)


print("== (a) bootstrap half-width of the paired per-feature AUC wedge ==")
for (d, k, ntest) in [(128, 16, 4000), (128, 16, 1000), (256, 32, 4000)]:
    g = torch.Generator(device=DEV).manual_seed(999 + d + 31 * k)
    m = 4096
    F = make_dict(d, m, g)
    Wd, bd = linear_detector(F, k, 20000, g, 0.01)
    xt, at = sample(F, k, ntest, g, 0.01)
    cs = fista(F, xt, 0.06 if k == 16 else 0.01)
    de = xt @ Wd.T + bd
    a_cs, ok = per_feature_auc_vec(cs, at)
    a_de, _ = per_feature_auc_vec(de, at)
    diff = (a_cs - a_de)[ok].cpu().numpy()
    B = 4000
    rng = np.random.default_rng(0)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"d={d} k={k} n_test={ntest}: n_features={len(diff)} "
          f"mean_wedge={diff.mean():+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
          f"half_width={(hi - lo) / 2:.4f}", flush=True)

print()
print("== (b) Route B literal test: can a LINEAR probe fit a non-linearly-separable")
print("      function of day-of-week when there are only 7 distinct stimuli? ==")
rng = np.random.default_rng(7)
for dim in [2, 6, 8, 64, 768]:
    # 7 'day' representations. Worst case for the wedge story: an exact 2-D circle.
    th = 2 * np.pi * np.arange(7) / 7
    if dim == 2:
        X = np.stack([np.cos(th), np.sin(th)], 1)
    else:
        U = rng.normal(size=(2, dim))
        X = np.stack([np.cos(th), np.sin(th)], 1) @ U + 0.0
    for name, y in [("parity t%2", (np.arange(7) % 2).astype(float)),
                    ("t mod 3",   (np.arange(7) % 3).astype(float)),
                    ("ordinal t", np.arange(7).astype(float))]:
        Xa = np.concatenate([X, np.ones((7, 1))], 1)
        w, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        resid = np.abs(Xa @ w - y).max()
        print(f"  dim={dim:4d} target={name:10s} max|residual| of best AFFINE probe = {resid:.2e}")
