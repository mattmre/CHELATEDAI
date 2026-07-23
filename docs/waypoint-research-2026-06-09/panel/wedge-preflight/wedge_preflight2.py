"""Preflight 2: (a) is the AUC wedge an artifact of an unfair linear baseline?
(b) how fast does the wedge die when the dictionary is only approximately known?"""
import torch
from wedge_preflight import make_dict, sample, fista, DEV

torch.manual_seed(0)


def per_feature_auc(scores, a):
    """Mean over features of the per-feature detection AUC (removes per-feature
    scale mismatch that pooled AUC unfairly punishes)."""
    lab = (a > 0).float()
    n, m = scores.shape
    order = torch.argsort(scores, dim=0)
    ranks = torch.empty_like(scores)
    ar = (torch.arange(n, device=DEV, dtype=scores.dtype) + 1).unsqueeze(1).expand(n, m)
    ranks.scatter_(0, order, ar)
    npos = lab.sum(0)
    nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    auc = ((ranks * lab).sum(0) - npos * (npos + 1) / 2) / (npos * nneg).clamp(min=1)
    return auc[ok].mean().item()


def pooled_auc(scores, a):
    s = scores.flatten()
    lab = (a > 0).float().flatten()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(len(s), device=DEV, dtype=s.dtype) + 1
    npos = lab.sum()
    nneg = len(s) - npos
    return (((ranks * lab).sum() - npos * (npos + 1) / 2) / (npos * nneg)).item()


def linear_detector(F, k, n_train, g, sigma):
    """Per-feature linear DETECTOR (ridge onto the binary active/inactive target)
    -- the fair strong linear baseline for a detection metric."""
    x, a = sample(F, k, n_train, g, sigma)
    y = (a > 0).float()
    d = F.shape[0]
    Sxx = (x.T @ x) / n_train
    Syx = (y.T @ x) / n_train
    W = torch.linalg.solve(Sxx + 1e-6 * torch.eye(d, device=DEV), Syx.T).T
    b = y.mean(0) - x.mean(0) @ W.T
    return W, b


def linear_mmse(F, k, n_train, g, sigma):
    x, a = sample(F, k, n_train, g, sigma)
    d = F.shape[0]
    Sxx = (x.T @ x) / n_train
    Sax = (a.T @ x) / n_train
    return torch.linalg.solve(Sxx + 1e-6 * torch.eye(d, device=DEV), Sax.T).T


def cell(d, k, m=4096, sigma=0.01, n_train=20000, n_test=4000, dict_err=0.0, lam=None):
    g = torch.Generator(device=DEV).manual_seed(999 + d + 31 * k)
    F = make_dict(d, m, g)
    Wm = linear_mmse(F, k, n_train, g, sigma)
    Wd, bd = linear_detector(F, k, n_train, g, sigma)
    xt, at = sample(F, k, n_test, g, sigma)

    if dict_err > 0:
        Fh = F + dict_err * torch.randn(F.shape, generator=g, device=DEV) / (d ** 0.5)
        Fh = Fh / Fh.norm(dim=0, keepdim=True)
    else:
        Fh = F
    coh = (F * Fh).sum(0).mean().item()  # mean per-atom cosine to truth

    if lam is None:
        xv, av = sample(F, k, 2000, g, sigma)
        best = (-1e9, 0.03)
        for L in [0.003, 0.01, 0.03, 0.06, 0.1, 0.2]:
            s = per_feature_auc(fista(Fh, xv, L), av)
            if s > best[0]:
                best = (s, L)
        lam = best[1]

    cs = fista(Fh, xt, lam)
    mm = xt @ Wm.T
    de = xt @ Wd.T + bd
    return dict(d=d, k=k, lam=lam, err=dict_err, coh=coh,
                mmse_pool=pooled_auc(mm, at), mmse_pf=per_feature_auc(mm, at),
                det_pf=per_feature_auc(de, at),
                cs_pool=pooled_auc(cs, at), cs_pf=per_feature_auc(cs, at))


print("== fair-baseline check (true dictionary) ==")
for (d, k) in [(64, 8), (128, 16), (256, 32), (128, 8), (256, 16), (512, 32)]:
    r = cell(d, k)
    print(f"d={r['d']:4d} k={r['k']:3d} lam={r['lam']:.3f} | "
          f"MMSE pooled={r['mmse_pool']:.3f} perfeat={r['mmse_pf']:.3f} | "
          f"DETECTOR perfeat={r['det_pf']:.3f} | CS pooled={r['cs_pool']:.3f} "
          f"perfeat={r['cs_pf']:.3f} | wedge_vs_detector={r['cs_pf']-r['det_pf']:+.4f}",
          flush=True)

print()
print("== dictionary-error sensitivity (d=128,k=16 and d=256,k=32) ==")
for (d, k) in [(128, 16), (256, 32)]:
    for e in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
        r = cell(d, k, dict_err=e)
        print(f"d={d} k={k} err={e:.2f} atom_cos={r['coh']:.3f} | "
              f"DET perfeat={r['det_pf']:.3f} CS perfeat={r['cs_pf']:.3f} "
              f"wedge={r['cs_pf']-r['det_pf']:+.4f}", flush=True)
