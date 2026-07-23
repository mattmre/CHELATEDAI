"""Generate the paper's 3 core figures from REAL measured numbers (this session's runs; no fabricated
values). Pristine layout: constrained_layout (no clipping/overlap), figure-centered titles that fit,
extended notes live in the markdown captions, not the image."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titlepad": 6,
                     "figure.dpi": 170, "savefig.dpi": 170})
OUT = r"D:\GITHUB\CHELATEDAI\docs\waypoint-research-2026-06-09\paper-draft\figures"
os.makedirs(OUT, exist_ok=True)

# ---------- Fig 1: recovery ladder (SciFact, % of oracle gap) ----------
ladder = [
    ("frozen floor", 0.0, "#bbbbbb"),
    ("bounded near-identity (α=0.1)", 2.1, "#ff9896"),
    ("anchor-InfoNCE corrector (C3a)", 20.0, "#d62728"),
    ("low-rank affine (r=64)", 48.7, "#9edae5"),
    ("least-squares (full)", 66.9, "#1f77b4"),
    ("S1 doc2query self-pairs", 80.0, "#2ca02c"),
    ("residual MLP (nonlinear)", 81.2, "#aec7e8"),
    ("orthogonal Procrustes", 82.6, "#7fcdbb"),
    ("ridge (best linear)", 84.4, "#17becf"),
    ("oracle (full re-embed)", 100.0, "#7f7f7f"),
]
labels = [a for a, _, _ in ladder]
vals = [b for _, b, _ in ladder]
cols = [c for _, _, c in ladder]
fig, ax = plt.subplots(figsize=(8.6, 4.3), constrained_layout=True)
y = list(range(len(labels)))
ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.4)
for i, v in enumerate(vals):
    ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("recovery of the oracle gap  (%)")
ax.set_xlim(0, 116)
ax.axvline(84.4, color="#17becf", ls="--", lw=0.9, alpha=0.7)
fig.suptitle("A one-line linear map beats the elaborate corrector ~4.2×   (SciFact, MiniLM→mpnet)",
             fontsize=10.5, y=1.04)
fig.savefig(os.path.join(OUT, "fig1_recovery_ladder.png"), bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2: bound -> recovery curve ----------
alpha = [0.05, 0.1, 0.25, 0.5, 1.0]
rec = [0.9, 2.1, 11.8, 58.7, 84.4]
fig, ax = plt.subplots(figsize=(6.6, 4.1), constrained_layout=True)
ax.axhline(20.0, color="#d62728", ls=":", lw=1.2, alpha=0.8, label="our anchor-InfoNCE corrector (C3a) = 20%")
ax.plot(alpha, rec, "o-", color="#1f77b4", lw=1.9, ms=6, label="shrunk linear map:  I + α(W−I)")
ax.axhline(84.4, color="#17becf", ls="--", lw=1.1, label="ridge (regularized, unbounded) = 84.4%")
for a, r in zip(alpha, rec):
    ax.annotate(f"{r:.0f}%", (a, r), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8)
ax.set_xlabel("correction magnitude α   (α→0: near-identity bound  —  α=1: unbounded)")
ax.set_ylabel("recovery of oracle gap (%)")
ax.set_ylim(0, 95)
ax.legend(fontsize=8, loc="center right")
fig.suptitle("The near-identity bound is the secondary cause", fontsize=10.5, y=1.03)
fig.savefig(os.path.join(OUT, "fig2_bound_curve.png"), bbox_inches="tight")
plt.close(fig)

# ---------- Fig 3: generality ----------
groups = ["SciFact\nmpnet", "NFCorpus\nmpnet", "SciFact\nbge-large", "NFCorpus\nbge-large"]
ridge_g = [84.4, 66.4, 78.9, 74.3]
bound_g = [2.1, 2.2, 0.0, 2.1]
x = list(range(len(groups))); w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 4.1), constrained_layout=True)
b1 = ax.bar([i - w / 2 for i in x], ridge_g, w, label="trivial ridge (unbounded)",
            color="#17becf", edgecolor="black", lw=0.4)
b2 = ax.bar([i + w / 2 for i in x], bound_g, w, label="near-identity bounded (α=0.1)",
            color="#ff9896", edgecolor="black", lw=0.4)
ax.bar_label(b1, fmt="%.0f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("recovery of oracle gap (%)"); ax.set_ylim(0, 98)
ax.legend(fontsize=8.5, loc="upper right")
fig.suptitle("The pattern generalizes across encoder families and datasets", fontsize=10.5, y=1.03)
fig.savefig(os.path.join(OUT, "fig3_generality.png"), bbox_inches="tight")
plt.close(fig)

print("wrote:", sorted(os.listdir(OUT)))
