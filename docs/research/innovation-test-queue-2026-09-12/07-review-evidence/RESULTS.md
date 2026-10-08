# IQ-07 review evidence — captured 2026-09-17T18:51:03Z

Host: python 3.11.9 | torch 2.5.1+cu121

These three scripts are hand-repaired transcriptions of the pasted Phase-0
harnesses (the paste itself is not runnable; see the packet's Y0). They are
REVIEW INSTRUMENTATION for the IQ-07 packet, not product code and not tests.
Re-run any of them directly to reproduce the numbers quoted in the packet.

## `verify_rhpc_cycle_detector.py`

```text
norm growth on the loop trace:
   t=0 |H|=1.000e+00
   t=1 |H|=9.974e-01
   t=2 |H|=9.903e-01
   t=3 |H|=9.946e-01
   t=4 |H|=9.992e-01
   t=5 |H|=1.003e+00

valid DAG alarms (want 0): []
k=3 loop ABCABC alarms (want >0): []
k=2 loop ABABAB alarms (README claims detected): []
k=6 loop (README claims detected): []

false-positive rate over 200 random 5-step DAGs (README claims 0.0%):
   0/200 = 0.0%
```

## `verify_chelation_sensitivity.py`

```text
as-shipped noise (50/1024 dims, sigma=12):
  oracle variance-ratio : {'0%': 0.3569, '2%': 0.4441, '5%': 1.0, '10%': 1.0}
  noisy-variance only   : {'0%': 0.3569, '2%': 0.4455, '5%': 1.0, '10%': 1.0}
  (pass gate is Prune_5% >= 0.95)

noise at sigma=1 (same magnitude as signal), everything else identical:
  oracle variance-ratio : {'0%': 0.9766, '2%': 0.9856, '5%': 1.0, '10%': 1.0}

non-axis-aligned noise (dense, rotated basis), sigma=0.5:
  oracle variance-ratio : {'0%': 0.8945, '2%': 0.8945, '5%': 0.8945, '10%': 0.8946}
```

## `verify_coalition_probe.py`

```text
as-shipped (n=2000,d=1024, in-sample, real AND labels):
   linear 0.6445  bilinear 0.6810  base_rate 0.248
as-shipped params: linear X=2048 cols vs 2000 rows; bilinear X=3072 cols vs 2000 rows

SAME CODE, RANDOM LABELS (should fail if test measures anything):
   linear 0.6335  bilinear 0.6560  base_rate 0.259

held-out split (train 1000 / test 1000), real AND labels:
   linear 0.5300  bilinear 0.5530  base_rate 0.247

honest regime n=20000, d=64, held-out, real AND labels:
   linear 0.7567  bilinear 0.8340  base_rate 0.246
```

