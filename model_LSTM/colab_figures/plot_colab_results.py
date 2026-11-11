"""Figures for the tuned LSTM run in colab_results/.

1. Health indicator vs predicted RUL on the 100 test engines, laid out like section 8 of main.ipynb.
2. Optuna optimization history: validation RMSE per trial and the best-so-far curve.
3. Hyperparameter importance (fANOVA) for the same study.

Run from anywhere:  python model_LSTM/colab_figures/plot_colab_results.py
Saves PNGs next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                   # repo root (for Features/)
RESULTS = HERE.parent / "colab_results"     # model_LSTM/colab_results

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, ENGINE_LINE = "#e1e0d9", "#c3c2b7", "#b5b4ad"
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

# ---------------------------------------------------------------- 1. health vs predicted RUL
# Same layout as section 8 of main.ipynb. Left: each test engine's health trajectory, placed on the
# "flights before retirement" axis using its predicted RUL, against the training-fleet median.
# Right: health at the last observed flight vs predicted RUL.
pred = pd.read_csv(RESULTS / "lstm_tuned_test_predictions.csv")
test = pd.read_csv(ROOT / "Features" / "features_test.csv")[["EngineID", "Cycle", "health"]]
train = pd.read_csv(ROOT / "Features" / "features_train.csv")[["EngineID", "RUL", "health"]]

test = test.merge(pred, on="EngineID", validate="many_to_one")
assert test["EngineID"].nunique() == len(pred) == 100
test["RUL_est"] = test["RUL_pred"] + test.groupby("EngineID")["Cycle"].transform("max") - test["Cycle"]
last = test.loc[test.groupby("EngineID")["Cycle"].idxmax()]

HORIZON = 250
wide = test[test["RUL_est"] <= HORIZON].pivot(index="RUL_est", columns="EngineID", values="health")
fleet = train[train["RUL"] <= HORIZON].groupby("RUL")["health"].median()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", width_ratios=[1.5, 1])
ax = axes[0]
for _, eng in wide.items():
    eng = eng.dropna()
    ax.plot(-eng.index, eng.to_numpy(), color=ENGINE_LINE, linewidth=0.4, alpha=0.5)
ax.plot(-fleet.index, fleet.to_numpy(), color=BLUE, linewidth=2, label="Training fleet median")
ax.plot([], [], color=ENGINE_LINE, linewidth=1, label="Test engines (placed by predicted RUL)")
ax.set(title="Test health indicator vs predicted flights before retirement",
       xlabel="Predicted flights before retirement",
       ylabel="PC1, smoothed (higher = more degraded)")
ax.legend(loc="upper left")

rho = last["health"].corr(last["RUL_pred"], method="spearman")
ax = axes[1]
ax.scatter(last["health"], last["RUL_pred"], s=30, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
ax.set(title="Indicator at last observed flight", xlabel="PC1, smoothed", ylabel="Predicted RUL (flights)")
ax.set_ylim(bottom=0)
ax.text(0.98, 0.97, f"{len(last)} engines, Spearman ρ = {rho:.2f}", transform=ax.transAxes,
        ha="right", va="top", color=INK_2)
fig.savefig(HERE / "01_health_vs_predicted_rul.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- study
study = optuna.load_study(study_name="lstm_rul", storage=f"sqlite:///{(RESULTS / 'optuna_lstm.db').as_posix()}")
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

# ---------------------------------------------------------------- 2. optimization history
num = np.array([t.number for t in done])
val = np.array([t.value for t in done])
best = np.minimum.accumulate(val)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(num, val, s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Completed trial")
ax.step(num, best, where="post", color=ORANGE, linewidth=2, zorder=4, label="Best so far")
ax.set_title(f"Optuna optimization history ({len(done)} completed, {len(pruned)} pruned)")
ax.set_xlabel("Trial number")
ax.set_ylabel("Validation RMSE (cycles)")
ax.set_xlim(-1, len(study.trials))
ax.annotate(f"best {study.best_value:.2f} (trial {study.best_trial.number})",
            xy=(study.best_trial.number, study.best_value), xytext=(6, 6), textcoords="offset points",
            ha="left", va="bottom", color=INK_2)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "02_optimization_history.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3. hyperparameter importance
imp = optuna.importance.get_param_importances(
    study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
names, scores = list(imp.keys())[::-1], list(imp.values())[::-1]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(names, scores, height=0.6, color=BLUE)
for b, s in zip(bars, scores):
    ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{s:.2f}",
            va="center", color=INK_2)
ax.set_title("Hyperparameter importance for validation RMSE (fANOVA)")
ax.set_xlabel("Importance (share of variance explained)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(scores) * 1.15)
fig.tight_layout()
fig.savefig(HERE / "03_hyperparameter_importance.png", dpi=150)
plt.close(fig)

print("Best params:", study.best_params)
print("Importances:", {k: round(v, 3) for k, v in imp.items()})
