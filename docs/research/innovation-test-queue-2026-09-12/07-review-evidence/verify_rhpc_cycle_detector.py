import torch
# hand-repaired transcription of harness_rhpc.TrajectoryFalsifier (markdown mangling fixed)
class TF:
    def __init__(self, dim=2048, seed=42):
        torch.manual_seed(seed); self.dim=dim
        q,_ = torch.linalg.qr(torch.randn(dim,dim)); self.R=q
    def cc(self,x,y): return torch.fft.ifft(torch.fft.fft(x)*torch.fft.fft(y)).real
    def run(self, trace, window=6, threshold=0.88, verbose=False):
        hist=[]; H=torch.zeros(self.dim); alarms=[]
        for t,a in enumerate(trace):
            H = a if t==0 else self.cc(torch.matmul(self.R,H), a)
            if verbose: print(f"   t={t} |H|={H.norm().item():.3e}")
            for off,pH in enumerate(reversed(hist[-window:]), start=1):
                s=torch.cosine_similarity(H.unsqueeze(0),pH.unsqueeze(0)).item()
                if s>threshold: alarms.append({"step":t,"period":off,"sim":round(s,4)}); break
            hist.append(H)
        return alarms

tf=TF(2048)
A={k:torch.nn.functional.normalize(torch.randn(2048),dim=0) for k in "ABCDE"}
dag=[A[k] for k in "ABCDE"]
loop=[A[k] for k in "ABCABC"]
print("norm growth on the loop trace:"); tf.run(loop, verbose=True)
print()
print("valid DAG alarms (want 0):", tf.run(dag))
print("k=3 loop ABCABC alarms (want >0):", tf.run(loop))
print("k=2 loop ABABAB alarms (README claims detected):", tf.run([A[k] for k in "ABABAB"]))
print("k=6 loop (README claims detected):", tf.run([A[k] for k in "ABCDEABCDEABCDE"[:12]]))
print()
print("false-positive rate over 200 random 5-step DAGs (README claims 0.0%):")
fp=0
for s in range(200):
    torch.manual_seed(1000+s)
    tr=[torch.nn.functional.normalize(torch.randn(2048),dim=0) for _ in range(5)]
    if tf.run(tr): fp+=1
print(f"   {fp}/200 = {fp/2:.1f}%")
