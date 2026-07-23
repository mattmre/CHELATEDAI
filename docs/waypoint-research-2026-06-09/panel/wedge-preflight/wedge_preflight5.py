"""Preflight 5: the FAIR Route B linear baseline (ridge chosen on held-out contexts),
versus the pure-circle Fourier ceiling and the 2-D nonlinear readout."""
import numpy as np
from wedge_preflight4 import build, TARGETS, CIRC


def ridge_fit(X, y, lam):
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def rmse(X, y, w):
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    return float(np.sqrt(((A @ w - y) ** 2).mean()))


# analytic ceiling: best affine function of the EXACT rank-2 circle
print("pure-circle (rank-2) ceiling for the best affine readout:")
ceil = {}
for name, t in TARGETS.items():
    A = np.concatenate([CIRC, np.ones((7, 1))], 1)
    w, *_ = np.linalg.lstsq(A, t, rcond=None)
    ceil[name] = float(np.sqrt(((A @ w - t) ** 2).mean()))
    print(f"  {name:10s} RMSE_ceiling = {ceil[name]:.4f}  (target sd = {t.std():.4f})")

print()
print("fair linear probe: ridge selected on held-out VAL contexts, scored on TEST contexts")
for off in [0.05, 0.2]:
    for n_ctx in [20, 100]:
        Xtr, ytr = build(n_ctx, off, 21)
        Xva, yva = build(30, off, 55)
        Xte, yte = build(50, off, 99)
        for name in TARGETS:
            best = (1e9, None, None)
            for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100, 1000]:
                w = ridge_fit(Xtr, ytr[name], lam)
                v = rmse(Xva, yva[name], w)
                if v < best[0]:
                    best = (v, lam, w)
            te = rmse(Xte, yte[name], best[2])
            print(f"  off={off:.2f} n_ctx={n_ctx:3d} {name:10s} lam={best[1]:<8g} "
                  f"TEST RMSE={te:.4f}  ceiling={ceil[name]:.4f}  "
                  f"nonlinear atan2 RMSE=0.0000")
