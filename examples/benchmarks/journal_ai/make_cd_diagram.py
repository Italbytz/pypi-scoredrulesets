"""Generate a Demsar critical-difference (CD) diagram for the regression benchmark.

The average ranks are recomputed from the per-dataset out-of-sample R^2 values of
Table~\\ref{tab:regression_benchmark} (5 datasets, 7 methods). The Nemenyi critical
difference at alpha=0.05 for k=7 methods and N=5 datasets is CD = 4.03. Methods whose
average ranks differ by at most CD are connected by a thick clique bar.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

# Per-dataset out-of-sample R^2 (rows = datasets, cols = methods), from
# experiments/results_regression/regression_benchmark_results.csv and
# rulefit_benchmark_results.csv, as tabulated in Table tab:regression_benchmark.
METHODS = ["CART (Full)", "CART (d=4)", "ruleGP", "ruleNSGA-II", "rulePLCS", "RuleFit"]
R2 = np.array([
    [0.443, 0.535, 0.466, 0.463, 0.219, 0.654],   # California Housing
    [-0.137, 0.310, 0.363, 0.341, -3.647, 0.438],  # Diabetes
    [0.532, 0.571, 0.461, 0.431, -5.283, 0.753],   # Friedman #1
    [0.968, 0.917, 0.787, 0.743, 0.265, 0.955],    # Friedman #2
    [0.671, 0.650, 0.556, 0.631, -4.216, -0.021],  # Friedman #3
])
CD = 3.37  # Nemenyi critical difference, k=6, N=5, alpha=0.05

# Ranks per dataset (1 = best = highest R^2); average across datasets.
ranks = np.array([rankdata(-row, method="average") for row in R2])
avg_ranks = ranks.mean(axis=0)

order = np.argsort(avg_ranks)
names = [METHODS[i] for i in order]
ar = avg_ranks[order]

# Cliques: maximal sets of consecutive methods whose rank span <= CD.
cliques = []
for i in range(len(ar)):
    j = i
    while j + 1 < len(ar) and ar[j + 1] - ar[i] <= CD:
        j += 1
    if j > i:
        cliques.append((i, j))
# Drop cliques fully contained in another.
cliques = [c for c in cliques if not any(c != d and d[0] <= c[0] and c[1] <= d[1] for d in cliques)]

low, high = 1, 6  # rank axis bounds
fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.set_xlim(low - 0.3, high + 0.3)
ax.set_ylim(0, 1)
ax.axis("off")

# Top axis with rank ticks (best rank on the left).
y_axis = 0.82
ax.plot([low, high], [y_axis, y_axis], "k-", lw=1.2)
for r in range(low, high + 1):
    ax.plot([r, r], [y_axis, y_axis + 0.03], "k-", lw=1.2)
    ax.text(r, y_axis + 0.06, str(r), ha="center", va="bottom", fontsize=10)
ax.text((low + high) / 2, y_axis + 0.13, "Average rank", ha="center", va="bottom", fontsize=10)

# Split methods left/right for readable labels.
n = len(names)
left_idx = list(range(0, (n + 1) // 2))
right_idx = list(range((n + 1) // 2, n))

def elbow(x_rank, y_label, x_label):
    ax.plot([x_rank, x_rank], [y_axis, y_label], "k-", lw=1.0)
    ax.plot([x_rank, x_label], [y_label, y_label], "k-", lw=1.0)

for k, idx in enumerate(left_idx):
    y_label = y_axis - 0.12 - 0.11 * k
    elbow(ar[idx], y_label, low - 0.25)
    ax.text(low - 0.30, y_label, f"{names[idx]} ({ar[idx]:.2f})", ha="right", va="center", fontsize=10)

for k, idx in enumerate(right_idx):
    y_label = y_axis - 0.12 - 0.11 * (len(right_idx) - 1 - k)
    elbow(ar[idx], y_label, high + 0.25)
    ax.text(high + 0.30, y_label, f"{names[idx]} ({ar[idx]:.2f})", ha="left", va="center", fontsize=10)

# Clique bars.
bar_y = y_axis - 0.05
for c, (i, j) in enumerate(cliques):
    yb = bar_y - 0.028 * c
    ax.plot([ar[i] - 0.03, ar[j] + 0.03], [yb, yb], "k-", lw=4.0, solid_capstyle="round")

# CD reference bar.
cd_y = y_axis + 0.22
ax.plot([low, low + CD], [cd_y, cd_y], "k-", lw=1.4)
ax.plot([low, low], [cd_y - 0.02, cd_y + 0.02], "k-", lw=1.4)
ax.plot([low + CD, low + CD], [cd_y - 0.02, cd_y + 0.02], "k-", lw=1.4)
ax.text(low + CD / 2, cd_y + 0.03, f"CD = {CD:.2f}", ha="center", va="bottom", fontsize=10)

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"regression_cd_diagram.{ext}", bbox_inches="tight", dpi=200)
print("avg ranks:", {METHODS[i]: round(float(avg_ranks[i]), 2) for i in range(len(METHODS))})
print("cliques (by sorted index):", cliques)
print("saved to", out / "regression_cd_diagram.pdf")
