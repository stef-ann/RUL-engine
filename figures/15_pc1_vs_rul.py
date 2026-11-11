"""PC1 against RUL at three processing stages, on the 52 validation engines.

1. Unnormalized: PCA on all 21 sensors, standardized over the whole fleet (regime left in).
2. Per flight: PC1 of the 6 regime-normalized, Fisher-selected sensors, one point per flight.
3. Smoothed: the same PC1 after the 15-flight trailing average, i.e. the `health` feature.

Stages 2 and 3 repeat the fitting in features.ipynb (same split, baseline, Fisher selection and
PCA, all fitted on the training engines), and the script checks that stage 3 matches `health`
in Features/features_val.csv. Saves 15_pc1_vs_rul.png next to this script.

Run from anywhere:  python figures/15_pc1_vs_rul.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Same settings as features.ipynb
HEALTHY_FLIGHTS, LATE_FLIGHTS, SMOOTH_FLIGHTS, VAL_ENGINES = 30, 30, 15, 52
SPEEDS = ["Nf", "NRf", "Nc", "NRc"]

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.titleweight": "semibold",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})

# Load and split exactly as features.ipynb does
df = pd.read_csv(ROOT / "TrainingData_Engine.csv", encoding="utf-8")
df["RUL"] = df.groupby("EngineID")["Cycle"].transform("max") - df["Cycle"]
names = {c: c.split()[0] for c in df.columns[6:] if c != "RUL"}
sensors = list(names.values())
data = df.rename(columns=names)

val_ids = np.random.default_rng(0).choice(data["EngineID"].unique(), VAL_ENGINES, replace=False)
val = data[data["EngineID"].isin(val_ids)].reset_index(drop=True)
train = data[~data["EngineID"].isin(val_ids)].reset_index(drop=True)


def oriented_pc1(pca, train_x, x):
    """PC1 scores of x, signed so that higher means more degraded (lower RUL) on the training engines."""
    sign = -np.sign(np.corrcoef(pca.transform(train_x)[:, 0], train["RUL"])[0, 1])
    return sign * pca.transform(x)[:, 0]


# Stage 1: unnormalized, all sensors standardized over the whole training fleet
mean, std = train[sensors].mean(), train[sensors].std().replace(0, np.inf)
train_global = (train[sensors] - mean) / std
pca_raw = PCA(n_components=1).fit(train_global)
val["pc1_raw"] = oriented_pc1(pca_raw, train_global, (val[sensors] - mean) / std)

# Stage 2: regime-normalized against healthy flights, Fisher-selected sensors
baseline = train[train["Cycle"] <= HEALTHY_FLIGHTS].groupby("Regime")[sensors].agg(["mean", "std"])


def normalize(frame):
    mu = baseline.xs("mean", axis=1, level=1).reindex(frame["Regime"]).to_numpy()
    sd = baseline.xs("std", axis=1, level=1).reindex(frame["Regime"]).to_numpy()
    return pd.DataFrame((frame[sensors].to_numpy() - mu) / np.where(sd > 1e-9, sd, np.inf), columns=sensors, index=frame.index)


z_train, z_val = normalize(train), normalize(val)
healthy, late = z_train[train["Cycle"] <= HEALTHY_FLIGHTS], z_train[train["RUL"] < LATE_FLIGHTS]
fisher = ((late.mean() - healthy.mean()) ** 2 / (late.var() + healthy.var())).fillna(0)
selected = [s for s in fisher.sort_values(ascending=False).index[fisher.sort_values(ascending=False) >= 2] if s not in SPEEDS]
pca = PCA().fit(z_train[selected])
val["pc1"] = oriented_pc1(pca, z_train[selected], z_val[selected])

# Stage 3: 15-flight trailing average per engine
val["health"] = val.groupby("EngineID")["pc1"].transform(lambda s: s.rolling(SMOOTH_FLIGHTS).mean())

saved = pd.read_csv(ROOT / "Features" / "features_val.csv")
check = saved.merge(val[["EngineID", "Cycle", "health"]], on=["EngineID", "Cycle"], suffixes=("_saved", ""))
assert len(check) == len(saved) and np.allclose(check["health_saved"], check["health"]), "Does not match features.ipynb"

# Plot every validation flight where the smoothed value exists, so all panels use the same rows
rows = val[val["Cycle"] >= SMOOTH_FLIGHTS]
panels = [
    ("pc1_raw", "1. Unnormalized PC1, all 21 sensors", "Regime differences dominate"),
    ("pc1", f"2. Regime-normalized PC1, {len(selected)} sensors", "One point per flight"),
    ("health", "3. Smoothed PC1 (health)", f"{SMOOTH_FLIGHTS}-flight trailing average"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, layout="constrained")
for ax, (column, title, note) in zip(axes, panels):
    rho = rows[column].corr(rows["RUL"], method="spearman")
    ax.scatter(rows["RUL"], rows[column], s=2, color=BLUE, alpha=0.18, linewidths=0, rasterized=True)
    if column != "pc1_raw":   # a median across regime bands is meaningless, so skip it on panel 1
        median = rows[rows["RUL"] <= 250].groupby("RUL")[column].median()
        ax.plot(median.index, median.to_numpy(), color=ORANGE, linewidth=1.8, label="Median by RUL")
    ax.set_title(f"{title}\n{note}, Spearman ρ = {rho:+.2f}", fontsize=9.5)
    ax.set(xlabel="RUL (flights)", ylabel="PC1 (higher = more degraded)" if ax is axes[0] else None)
    ax.invert_xaxis()   # time runs left to right, retirement at the right edge
    print(f"{column:8s} Spearman with RUL {rho:+.3f}")
axes[1].sharey(axes[2])
axes[1].legend(loc="upper left")
fig.suptitle(f"PC1 against RUL, {val['EngineID'].nunique()} validation engines (not used for fitting)",
             x=0.01, ha="left", fontsize=10.5, fontweight="semibold")

out = HERE / "15_pc1_vs_rul.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
