"""Spearman correlation matrix of the feature vector and RUL on the validation engines.

Reads Features/features_val.csv (run features.ipynb first) and saves 14_spearman_matrix.png
next to this script. Validation engines were not used to fit the regime baseline, Fisher
selection or PCA, so these correlations are out-of-sample.

Features from the PCA (orange labels): `health` (trailing 15-flight mean of PC1),
`health_delta` (health minus the engine's first-20-flight PC1 mean) and `health_slope`
(slope of PC1 over the last 20 flights). The 6 sensors are regime-normalized and smoothed
only, and Cycle and RUL are counts.

Run from anywhere:  python figures/14_spearman_matrix.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SURFACE, INK, INK_2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
ORANGE = "#eb6834"
FROM_PCA = {"health", "health_delta", "health_slope"}   # all three are computed from PC1 in features.ipynb
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.titleweight": "semibold",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
})

features = pd.read_csv(ROOT / "Features" / "features_val.csv")
columns = [c for c in features.columns if c not in ("EngineID", "Cycle", "RUL")] + ["Cycle", "RUL"]
corr = features[columns].corr(method="spearman")

# Show the lower triangle only, so each pair appears once and the RUL row reads across the bottom.
lower = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1))

diverging = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
diverging.set_bad(SURFACE)

fig, ax = plt.subplots(figsize=(7.6, 6.6), layout="constrained")
image = ax.imshow(lower, cmap=diverging, vmin=-1, vmax=1)
ax.set_xticks(range(len(columns) - 1), columns[:-1], rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(1, len(columns)), columns[1:])
ax.set_xlim(-0.5, len(columns) - 1.5)
ax.set_ylim(len(columns) - 0.5, 0.5)
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
ax.set_title(f"Spearman correlation of features and RUL, {features['EngineID'].nunique()} validation engines")

for i in range(1, len(columns)):
    for j in range(i):
        value = lower.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(value) >= 0.6 else INK,
                fontweight="semibold" if columns[i] == "RUL" else "normal")

# Mark which features come from the PCA. The 6 sensors are only regime-normalized and smoothed.
for label in ax.get_xticklabels() + ax.get_yticklabels():
    if label.get_text() in FROM_PCA:
        label.set_color(ORANGE)
        label.set_fontweight("semibold")
ax.legend(
    handles=[Patch(color=ORANGE, label="From PCA (built from PC1)"),
             Patch(color=INK_2, label="Not from PCA (sensors, Cycle, RUL)")],
    loc="upper right", frameon=False, fontsize=9, handlelength=1, handleheight=1,
    labelcolor=[ORANGE, INK_2],
)

fig.colorbar(image, ax=ax, shrink=0.7, label="Spearman ρ").outline.set_visible(False)

out = HERE / "14_spearman_matrix.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
print(corr["RUL"].drop("RUL").sort_values().round(3).to_string())
