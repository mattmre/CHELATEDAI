import json, time, sys
import numpy as np
sys.path.insert(0, r"C:\Users\mattm\AppData\Local\Temp\claude\D--GITHUB-CHELATEDAI--claude-worktrees-relaxed-wozniak-271e04\00b9ffaa-9115-4aad-9219-bcc10f287c90\scratchpad")
from wedge_pilot import run_cell  # noqa

CELLS = [
    (512, 4096, 8, 0.0),    # linear-sufficient control  d/k^2 ln m = 0.96
    (1024, 4096, 8, 0.0),   # deep linear-sufficient control d/k^2 ln m = 1.92
    (64, 2048, 5, 0.0),     # small wedge cell
]
out = []
for (d, m, k, sg) in CELLS:
    t0 = time.time()
    r = run_cell(d, m, k, sigma=sg, n_test=1000, seed=1)
    df, de = r.pop("d_frac"), r.pop("d_exact")
    r["frac_diff_mean"] = float(df.mean()); r["frac_diff_sd"] = float(df.std(ddof=1))
    r["exact_diff_mean"] = float(de.mean()); r["exact_diff_sd"] = float(de.std(ddof=1))
    r["secs"] = round(time.time() - t0, 1)
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
