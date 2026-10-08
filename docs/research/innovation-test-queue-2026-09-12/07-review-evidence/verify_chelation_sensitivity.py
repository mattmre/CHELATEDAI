import torch
def evaluate(clean, noisy, rates=(0.0,0.02,0.05,0.10), oracle=True):
    dim=clean.shape[1]
    if oracle:
        score = (torch.var(noisy,0)+1e-8)/(torch.var(clean,0)+1e-8)   # needs clean == oracle
    else:
        score = torch.var(noisy,0)                                    # noisy-only, deployable
    out={}
    for r in rates:
        k=int(dim*r); m=torch.ones(dim)
        if k>0: m[torch.topk(score,k=k).indices]=0.
        a=torch.nn.functional.normalize(clean*m,dim=1); b=torch.nn.functional.normalize(noisy*m,dim=1)
        out[f"{int(r*100)}%"]=round((a*b).sum(1).mean().item(),4)
    return out
torch.manual_seed(42)
N,D=1000,1024
clean=torch.randn(N,D); cor=clean.clone()
nd=torch.randperm(D)[:50]; cor[:,nd]+=torch.randn(N,50)*12.0
print("as-shipped noise (50/1024 dims, sigma=12):")
print("  oracle variance-ratio :", evaluate(clean,cor))
print("  noisy-variance only   :", evaluate(clean,cor,oracle=False))
print("  (pass gate is Prune_5% >= 0.95)")
print()
cor2=clean.clone(); cor2[:,nd]+=torch.randn(N,50)*1.0
print("noise at sigma=1 (same magnitude as signal), everything else identical:")
print("  oracle variance-ratio :", evaluate(clean,cor2))
print()
cor3=clean.clone()
Rq,_=torch.linalg.qr(torch.randn(D,D))
cor3 = clean + (torch.randn(N,D)@Rq)*0.5   # non-axis-aligned (rotated) noise
print("non-axis-aligned noise (dense, rotated basis), sigma=0.5:")
print("  oracle variance-ratio :", evaluate(clean,cor3))
