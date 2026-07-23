"""Preflight 4 (Route B design): the finite-stimulus universality trap, and the
held-out-context fix. Synthetic mimic of a circular day-of-week feature living in
a real encoder's space, with context-specific off-plane variance."""
import numpy as np

rng = np.random.default_rng(11)
DIM = 768
TH = 2 * np.pi * np.arange(7) / 7
U = rng.normal(size=(2, DIM)) / np.sqrt(DIM)
CIRC = np.stack([np.cos(TH), np.sin(TH)], 1) @ U  # 7 x DIM, exactly rank 2

TARGETS = {"parity t%2": (np.arange(7) % 2).astype(float),
           "t mod 3": (np.arange(7) % 3).astype(float),
           "ordinal t": np.arange(7).astype(float)}


def build(n_ctx, off_plane, seed):
    """x(ctx, day) = circle(day) + per-context offset + PER-ITEM off-plane noise.
    The per-item term is what makes a finite stimulus set affinely independent."""
    r = np.random.default_rng(seed)
    ctx = r.normal(size=(n_ctx, DIM)) / np.sqrt(DIM) * off_plane
    item = r.normal(size=(n_ctx, 7, DIM)) / np.sqrt(DIM) * off_plane
    X = (CIRC[None, :, :] + ctx[:, None, :] + item).reshape(n_ctx * 7, DIM)
    y = {k: np.tile(v, n_ctx) for k, v in TARGETS.items()}
    return X, y


def affine_fit_eval(Xtr, ytr, Xte, yte, ridge=1e-8):
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ ytr)
    B = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    return np.abs(A @ w - ytr).max(), np.abs(B @ w - yte).max()


print("== trap: 7 distinct stimuli only, probe fit and evaluated on the SAME 7 ==")
for off in [0.0, 0.01, 0.1]:
    X, _ = build(1, off, 3)
    for name, t in TARGETS.items():
        ins, _ = affine_fit_eval(X, t, X, t)
        print(f"  off_plane={off:.2f} target={name:10s} in-sample max|resid| = {ins:.2e}")

print()
print("== fix: many contexts, fit on TRAIN contexts, evaluate on HELD-OUT contexts ==")
for off in [0.05, 0.2]:
    for n_ctx in [5, 20, 100]:
        Xtr, ytr = build(n_ctx, off, 21)
        Xte, yte = build(50, off, 99)
        for name in TARGETS:
            ins, oos = affine_fit_eval(Xtr, ytr[name], Xte, yte[name])
            print(f"  off={off:.2f} n_ctx={n_ctx:4d} target={name:10s} "
                  f"in={ins:.2e} HELD-OUT max|resid|={oos:.3f}")
    print()

print("== nonlinear 2-D readout (recover plane by PCA on class means, then atan2) ==")
for off in [0.05, 0.2]:
    Xtr, _ = build(20, off, 21)
    M = Xtr.reshape(20, 7, DIM).mean(0)
    Mc = M - M.mean(0)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    P = Vt[:2]
    Xte, _ = build(50, off, 99)
    Z = (Xte - M.mean(0)) @ P.T
    ang = np.arctan2(Z[:, 1], Z[:, 0])
    ref = np.arctan2((Mc @ P.T)[:, 1], (Mc @ P.T)[:, 0])
    pred = np.argmin(np.abs(np.angle(np.exp(1j * (ang[:, None] - ref[None, :])))), 1)
    true = np.tile(np.arange(7), 50)
    print(f"  off={off:.2f} held-out day recovery accuracy from atan2 readout = "
          f"{(pred == true).mean():.4f}  "
          f"(=> exact recovery of ANY function of day, incl. parity / t mod 3)")
