"""Preflight 6: validate the FISTA solver used in the 'nonlinear' arm against
sklearn's Lasso reference. A bug here would corrupt every wedge number."""
import numpy as np
import torch
from sklearn.linear_model import Lasso
from wedge_preflight import make_dict, sample, fista, DEV

g = torch.Generator(device=DEV).manual_seed(5)
d, m, k = 64, 512, 8
F = make_dict(d, m, g)
x, a = sample(F, k, 20, g, 0.01)

for lam, nn in [(0.01, False), (0.06, False), (0.01, True), (0.06, True)]:
    # sklearn objective: 1/(2*n_samples)||y-Xw||^2 + alpha||w||_1, n_samples=d here
    alpha = lam / d
    Fn = F.cpu().numpy().astype(np.float64)
    mine = fista(F, x, lam, iters=5000, nonneg=nn).cpu().numpy().astype(np.float64)
    ref = np.stack([Lasso(alpha=alpha, fit_intercept=False, positive=nn, max_iter=200000, tol=1e-12)
                    .fit(Fn, xi).coef_ for xi in x.cpu().numpy().astype(np.float64)])

    def obj(A):
        return (0.5 * ((A @ Fn.T - x.cpu().numpy()) ** 2).sum(1) + lam * np.abs(A).sum(1))

    om, orf = obj(mine), obj(ref)
    print(f"lam={lam} nonneg={nn}: max|coef diff|={np.abs(mine - ref).max():.3e}  "
          f"mean obj mine={om.mean():.6f} sklearn={orf.mean():.6f}  "
          f"max(mine-sklearn) objective gap={np.max(om - orf):+.3e}")
