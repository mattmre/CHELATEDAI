import numpy as np
from scipy import stats

z975 = stats.norm.ppf(0.975)
z80 = stats.norm.ppf(0.80)
z95 = stats.norm.ppf(0.95)
print("z .975=%.4f  z .80=%.4f  z .95=%.4f" % (z975, z80, z95))

ns = [200, 1000, 10000, 100000]
sds = [0.13, 0.30, 0.50]
print("\nPaired-difference MDE  delta = (z975 + z_pow) * sd / sqrt(n)")
print("sd     n        hw95=1.96*sd/sqrt(n)  MDE80        MDE95")
for sd in sds:
    for n in ns:
        hw = z975 * sd / np.sqrt(n)
        m80 = (z975 + z80) * sd / np.sqrt(n)
        m95 = (z975 + z95) * sd / np.sqrt(n)
        print("%.2f   %-8d %-21.5f %-12.5f %.5f" % (sd, n, hw, m80, m95))

print("\nMcNemar exact: min discordant pairs for two-sided p<0.05 if all favour one arm")
for nd in range(3, 10):
    p = 2 * 0.5 ** nd
    print("  n_disc=%d  p=%.5f  %s" % (nd, p, "SIG" if p < 0.05 else ""))

za = stats.norm.ppf(1 - 0.05 / 12)
print("\nBonferroni over 6 cells (alpha=0.00833 two-sided), z_a/2=%.4f, sd=0.13" % za)
for n in ns:
    print("  n=%-7d MDE80=%.5f" % (n, (za + z80) * 0.13 / np.sqrt(n)))

print("\nCS vs linear dimension thresholds (nats)")
for (d, m, k) in [(128, 4096, 8), (96, 4096, 6), (256, 8192, 12), (64, 2048, 5), (512, 4096, 8), (1024, 4096, 8)]:
    cs = 2 * k * np.log(m / k)
    lin = k * k * np.log(m)
    print("  d=%-5d m=%-5d k=%-3d  2k ln(m/k)=%7.1f  k^2 ln m=%8.1f  d/cs=%5.2f  d/lin=%.2f"
          % (d, m, k, cs, lin, d / cs, d / lin))

print("\nDonoho-Tanner regime: delta=d/m, rho=k/d")
for (d, m, k) in [(128, 4096, 8), (96, 4096, 6), (256, 8192, 12), (64, 2048, 5), (512, 4096, 8)]:
    print("  d=%-5d m=%-5d k=%-3d  delta=%.4f  rho=%.4f" % (d, m, k, d / m, k / d))

print("\nLinear-probe interference heuristic (top-k selection):")
for (d, m, k) in [(128, 4096, 8), (96, 4096, 6), (256, 8192, 12), (512, 4096, 8)]:
    sig = np.sqrt(k / d)
    mx = sig * np.sqrt(2 * np.log(m - k))
    print("  d=%-5d m=%-5d k=%-3d  per-coord interference sd=%.3f  E[max over %d distractors]=%.3f  (signal=1.0)"
          % (d, m, k, sig, m - k, mx))
