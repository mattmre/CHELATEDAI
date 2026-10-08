import torch
torch.manual_seed(42)

def probe(n_samples, d_model, randomize_labels=False, split=False):
    h1 = torch.randn(n_samples, d_model); h2 = torch.randn(n_samples, d_model)
    f1 = (h1[:,0] > 0.).float(); f2 = (h2[:,0] > 0.).float()
    y = torch.logical_and(f1==1., f2==1.).float().unsqueeze(-1)
    if randomize_labels:
        y = (torch.rand(n_samples,1) > 0.75).float()   # match ~25% base rate
    def fit(X, y):
        if split:
            n = n_samples//2
            Xtr,ytr,Xte,yte = X[:n],y[:n],X[n:],y[n:]
        else:
            Xtr,ytr,Xte,yte = X,y,X,y
        w = torch.linalg.solve(Xtr.T@Xtr + 1e-3*torch.eye(X.shape[1]), Xtr.T@ytr)
        return ((torch.sigmoid(Xte@w) > .5).float() == yte).float().mean().item()
    lin = fit(torch.cat([h1,h2],-1), y)
    bi  = fit(torch.cat([h1,h2,h1*h2],-1), y)
    return lin, bi, y.mean().item()

print("as-shipped (n=2000,d=1024, in-sample, real AND labels):")
print("   linear %.4f  bilinear %.4f  base_rate %.3f" % probe(2000,1024))
print("as-shipped params: linear X=2048 cols vs 2000 rows; bilinear X=3072 cols vs 2000 rows")
print()
print("SAME CODE, RANDOM LABELS (should fail if test measures anything):")
print("   linear %.4f  bilinear %.4f  base_rate %.3f" % probe(2000,1024,randomize_labels=True))
print()
print("held-out split (train 1000 / test 1000), real AND labels:")
print("   linear %.4f  bilinear %.4f  base_rate %.3f" % probe(2000,1024,split=True))
print()
print("honest regime n=20000, d=64, held-out, real AND labels:")
print("   linear %.4f  bilinear %.4f  base_rate %.3f" % probe(20000,64,split=True))
