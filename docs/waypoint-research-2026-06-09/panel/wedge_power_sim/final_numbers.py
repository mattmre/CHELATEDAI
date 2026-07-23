import numpy as np
from scipy import stats

z975, z80 = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
zb = stats.norm.ppf(1 - 0.05 / 12)  # Bonferroni over 6 cells, two-sided
print("z975=%.4f z80=%.4f zBonf=%.4f" % (z975, z80, zb))

sd = 0.1284  # observed paired sd, operating cell d=128 m=4096 k=8 sigma=0.05
delta = 0.532625
print("\nOperating cell: sd_d=%.4f, pilot Delta=%.4f" % (sd, delta))
for n in [200, 1000, 10000, 100000]:
    hw = z975 * sd / np.sqrt(n)
    mde = (zb + z80) * sd / np.sqrt(n)
    print("  n=%-7d hw95=%.5f  MDE80(Bonf)=%.5f  Delta/MDE=%7.1fx  CI_low=%.4f"
          % (n, hw, mde, delta / mde, delta - hw))

print("\nNull/control cell d=1024: Delta=0.005 sd=0.024507")
for n in [1000, 10000]:
    hw = z975 * 0.024507 / np.sqrt(n)
    print("  n=%-6d hw95=%.5f  CI=[%.5f, %.5f]" % (n, hw, 0.005 - hw, 0.005 + hw))

print("\nProposed bar 0.15 -- headroom check")
for n in [1000, 10000]:
    mde = (zb + z80) * sd / np.sqrt(n)
    print("  n=%-6d bar/MDE80=%5.1fx   pilot CI_low=%.4f  clears bar by %.4f"
          % (n, 0.15 / mde, delta - z975 * sd / np.sqrt(n), delta - z975 * sd / np.sqrt(n) - 0.15))

print("\nFISTA cost model: flops = 4*d*m*n*iters")
d, m, iters = 128, 4096, 250
obs_n, obs_s = 1000, 68.4
f = 4 * d * m * obs_n * iters
rate = f / obs_s
print("  measured: n=1000 -> %.3e flops in %.1fs -> %.2f GFLOP/s effective (contended CPU)"
      % (f, obs_s, rate / 1e9))
for n in [10000, 100000]:
    fl = 4 * d * m * n * iters
    print("  n=%-7d flops=%.3e  CPU@%.1fGF/s -> %6.1f min | GPU@15TF/s -> %.1f s"
          % (n, fl, rate / 1e9, fl / rate / 60, fl / 15e12))

print("\nInterference prediction vs measurement (linear top-k readout)")
for (d, m, k, meas) in [(128, 4096, 8, 0.48625), (96, 4096, 6, 0.42283),
                        (256, 8192, 12, 0.483), (64, 2048, 5, 0.4758),
                        (512, 4096, 8, 0.920375), (1024, 4096, 8, 0.995)]:
    sig = np.sqrt(k / d) * np.sqrt(2 * np.log(m - k))
    print("  d=%-5d m=%-5d k=%-3d  E[max distractor]/signal=%.3f  measured lin_frac=%.4f"
          % (d, m, k, sig, meas))
