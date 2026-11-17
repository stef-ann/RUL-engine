# %% [markdown]
# # Bayesian RUL prediction (Group 3, Project 4)
#
# Method: Yan et al. (2016) / Gebraeel et al. (2005) style Bayesian degradation model.
#
#   1. Health indicator (HI): the same one as the group repo (features.ipynb):
#      regime-wise z-score vs. flights 1-30, six sensors, PCA PC1, signed so
#      higher = more degraded. We use the *unsmoothed* PC1 here, because the
#      Bayesian regression assumes independent noise per flight.
#   2. A curve shape for how HI grows over an engine's life. Two are compared:
#        quadratic:    HI(t) = th0 + th1*t + th2*t^2
#        exponential:  HI(t) = a + b*exp(c*(t - 2))
#      (t = flight number / 100; the "- 2" just keeps the numbers well scaled)
#   3. Prior: fit the curve to every training engine -> how the curve
#      parameters vary across the fleet. Noise variance pooled from residuals.
#   4. Failure threshold: fitted HI at each training engine's last flight
#      -> Normal(mean, sd).
#   5. For an engine observed up to flight n: Bayesian update of the prior with
#      its n HI values -> posterior of the curve parameters.
#   6. Monte Carlo: sample curves from the posterior and thresholds from the
#      threshold distribution, find where each curve first crosses its threshold.
#      RUL = crossing flight - n. Median = point estimate, 5-95% = interval.
#
# Validation follows the group split (52 engines, np.random.default_rng(0)) and
# scores every validation flight from flight 20 on, same rows as features_val.csv.
#
# Run as a script:  .venv\Scripts\python.exe bayesian_rul.py
# or cell by cell in VS Code (the "# %%" markers are cells).

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # running as a script: just save figures
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "bayesian_outputs"
OUT.mkdir(exist_ok=True)

# Same choices as the group repo (features.ipynb)
SENSORS = ["Ps30", "T50", "BPR", "htBleed", "T24", "T30"]  # Fisher >= 2, speed sensors excluded
HEALTHY_FLIGHTS = 30   # baseline flights for regime normalization
VAL_ENGINES = 52       # validation engines (about 20% of 260)
SPLIT_SEED = 0
EVAL_START = 20        # score flights from 20 on (same rows as features_val.csv)

# Bayesian model settings
TIME_SCALE = 100.0     # t = flight / 100
N_SAMPLES = 2000       # Monte Carlo samples per prediction
RUL_CAP = 500          # if a sampled curve never reaches the threshold
SEED = 42
EXP_T_REF = 2.0                          # exponential term is centred at flight 200
EXP_RATES = np.geomspace(0.3, 5.0, 70)                   # grid of growth rates c (exponential model)
EXP_LOG_B = np.linspace(np.log(0.2), np.log(400.0), 90)  # grid of log b (exponential model)


# %% Load data
def load_runs(path, key):
    """Turn the .mat struct array into one long table: one row per flight."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [str(s).split()[0] for s in m["Variable_List"]]
    frames = []
    for i, run in enumerate(np.atleast_1d(m[key]), start=1):
        df = pd.DataFrame(np.atleast_2d(run.Data), columns=names)
        df.insert(0, "Regime", np.atleast_1d(run.Regime).astype(int))
        df.insert(0, "Cycle", np.arange(1, len(df) + 1))
        df.insert(0, "EngineID", i)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


full = load_runs(ROOT / "Training_DataSet" / "TrainingData_Engine.mat", "Train_EngineRun")
test = load_runs(ROOT / "Testing_DataSet" / "TestingData_Engine.mat", "Testing_EngineRun")
full["RUL"] = full.groupby("EngineID")["Cycle"].transform("max") - full["Cycle"]

ids = full["EngineID"].unique()
val_ids = np.random.default_rng(SPLIT_SEED).choice(ids, VAL_ENGINES, replace=False)
is_val = full["EngineID"].isin(val_ids)
train, val = full[~is_val].copy(), full[is_val].copy()
assert not set(train.EngineID) & set(val.EngineID)

print(f"train: {train.EngineID.nunique()} engines, {len(train)} flights")
print(f"val:   {val.EngineID.nunique()} engines, {len(val)} flights")
print(f"test:  {test.EngineID.nunique()} engines, {len(test)} flights "
      f"(observed {test.groupby('EngineID').size().min()}-{test.groupby('EngineID').size().max()} flights)")


# %% Health indicator (fitted on training engines only)
base = train[train["Cycle"] <= HEALTHY_FLIGHTS].groupby("Regime")[SENSORS]
base_mu, base_sd = base.mean(), base.std()
base_sd = base_sd.where(base_sd > 1e-9, np.inf)  # constant sensor -> z = 0


def regime_z(df):
    mu = base_mu.loc[df["Regime"]].to_numpy()
    sd = base_sd.loc[df["Regime"]].to_numpy()
    return (df[SENSORS].to_numpy() - mu) / sd


pca = PCA().fit(regime_z(train))
pc1_sign = -np.sign(np.corrcoef(pca.transform(regime_z(train))[:, 0], train["RUL"])[0, 1])
for df in (train, val, test):
    df["HI"] = pc1_sign * pca.transform(regime_z(df))[:, 0]

print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance (group repo: 71.0%)")
print(f"corr(HI, RUL) on train: {np.corrcoef(train.HI, train.RUL)[0, 1]:+.2f}")
train_lives = train.groupby("EngineID")["Cycle"].max().to_numpy()


# %% The two Bayesian degradation models
class BayesianDegradationModel:
    """Shared machinery. Subclasses define the curve shape.

    Subclass methods:
      fit(train)                         -> learn the fleet prior, noise and threshold
      prefix_stats(cycles, hi)           -> running sums so flights 1..k can be used for any k
      sample_params(stats, k, n, rng)    -> n posterior samples of the curve parameters after flights 1..k+1
      curves(params, cycles)             -> HI curves for those parameter samples
      crossing_rul(params, D, cycle_now) -> flights until each curve reaches its threshold D
      fleet_curve(i, cycles)             -> fitted curve of training engine i (for plots)
    """

    def rul_samples(self, stats, k, cycle_now, rng, n=N_SAMPLES):
        params = self.sample_params(stats, k, n, rng)
        D = rng.normal(self.thr_mu, self.thr_sd, n)
        return np.minimum(self.crossing_rul(params, D, cycle_now), RUL_CAP)

    def predict_engine(self, g, eval_cycles, rng):
        """Predict RUL at each flight in eval_cycles using only flights 1..that flight."""
        stats = self.prefix_stats(g["Cycle"].to_numpy(), g["HI"].to_numpy())
        rows = []
        for c in eval_cycles:
            r = self.rul_samples(stats, c - 1, c, rng)
            q05, q50, q95 = np.percentile(r, [5, 50, 95])
            rows.append((c, q50, q05, q95))
        return pd.DataFrame(rows, columns=["Cycle", "pred", "lo90", "hi90"])

    def describe(self):
        print(f"[{self.name}] noise sd {np.sqrt(self.sigma2):.3f}, "
              f"failure threshold {self.thr_mu:.2f} +/- {self.thr_sd:.2f}")


def design_quad(cycles):
    t = np.asarray(cycles, dtype=float) / TIME_SCALE
    return np.column_stack([np.ones_like(t), t, t**2])


class QuadraticModel(BayesianDegradationModel):
    """HI(t) = th0 + th1 t + th2 t^2, Gaussian prior on (th0, th1, th2): exact conjugate update."""
    name = "quadratic"

    def fit(self, train):
        thetas, thresholds, sse, n_obs = [], [], 0.0, 0
        for _, g in train.groupby("EngineID"):
            X, y = design_quad(g["Cycle"]), g["HI"].to_numpy()
            th = np.linalg.lstsq(X, y, rcond=None)[0]
            thetas.append(th)
            thresholds.append(X[-1] @ th)  # fitted HI at retirement
            sse += np.sum((y - X @ th) ** 2)
            n_obs += len(y)
        self.fleet = np.array(thetas)
        self.mu0 = self.fleet.mean(axis=0)
        self.prec0 = np.linalg.inv(np.cov(self.fleet.T))
        self.sigma2 = sse / (n_obs - 3 * len(thetas))
        self.thr_mu, self.thr_sd = np.mean(thresholds), np.std(thresholds, ddof=1)
        return self

    def fleet_curve(self, i, cycles):
        return design_quad(cycles) @ self.fleet[i]

    def prefix_stats(self, cycles, hi):
        X = design_quad(cycles)
        return np.cumsum(np.einsum("ni,nj->nij", X, X), axis=0), np.cumsum(X * hi[:, None], axis=0)

    def sample_params(self, stats, k, n, rng):
        XtX, Xty = stats[0][k], stats[1][k]
        cov = np.linalg.inv(self.prec0 + XtX / self.sigma2)
        mu = cov @ (self.prec0 @ self.mu0 + Xty / self.sigma2)
        return mu + rng.standard_normal((n, 3)) @ np.linalg.cholesky(cov).T

    def curves(self, params, cycles):
        return params @ design_quad(cycles).T

    def crossing_rul(self, th, D, cycle_now):
        c, b, a = th[:, 0] - D, th[:, 1], th[:, 2]
        t0 = cycle_now / TIME_SCALE
        already = c + b * t0 + a * t0**2 >= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.sqrt(b * b - 4 * a * c)  # nan when no real root
            roots = np.stack([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
            linear = np.abs(a) < 1e-12
            roots[:, linear] = -c[linear] / b[linear]
        roots = np.where(np.isfinite(roots) & (roots > t0), roots, np.inf)
        rul = (roots.min(axis=0) - t0) * TIME_SCALE
        rul[already] = 0.0
        return rul


class ExponentialModel(BayesianDegradationModel):
    """HI(t) = a + b exp(c (t - 2)), with b > 0 and c > 0 (degradation only grows).

    b and c sit inside the non-linear part, so there is no closed-form update for them.
    Trick: try every (log b, c) pair on a grid. For a fixed pair the model is linear in a,
    so the exact Gaussian update works, and it also gives how well that pair explains the
    data (the marginal likelihood). Posterior weight of a pair = prior * likelihood.
    Prior: (a, log b, log c) jointly Gaussian across the training fleet.
    """
    name = "exponential"

    def basis(self, cycles):
        t = np.asarray(cycles, dtype=float) / TIME_SCALE
        return np.exp(np.outer(t - EXP_T_REF, EXP_RATES))  # (flights, rates)

    def fit(self, train):
        fits, thresholds, sse_total, n_obs = [], [], 0.0, 0
        for _, g in train.groupby("EngineID"):
            y, F = g["HI"].to_numpy(), self.basis(g["Cycle"])
            # least squares of y = a + b f, for every rate on the grid at once
            Fc, yc = F - F.mean(axis=0), y - y.mean()
            b = (Fc * yc[:, None]).sum(axis=0) / (Fc**2).sum(axis=0)
            a = y.mean() - b * F.mean(axis=0)
            sse = ((yc[:, None] - Fc * b) ** 2).sum(axis=0)
            j = np.argmin(sse)
            fits.append((a[j], b[j], EXP_RATES[j]))
            thresholds.append(a[j] + b[j] * F[-1, j])
            sse_total += sse[j]
            n_obs += len(y)
        self.fleet = np.array(fits)
        self.edge_share = np.mean(np.isin(self.fleet[:, 2], EXP_RATES[[0, -1]]))
        self.sigma2 = sse_total / (n_obs - 3 * len(fits))
        self.thr_mu, self.thr_sd = np.mean(thresholds), np.std(thresholds, ddof=1)
        assert (self.fleet[:, 1] > 0).all(), "a training engine has a decreasing exponential fit"

        # Joint Gaussian prior on z = (a, log b, log c)
        Z = np.column_stack([self.fleet[:, 0], np.log(self.fleet[:, 1:])])
        m, S = Z.mean(axis=0), np.cov(Z.T)
        Sbc_inv = np.linalg.inv(S[1:, 1:])
        grid = np.stack(np.meshgrid(EXP_LOG_B, np.log(EXP_RATES), indexing="ij"), axis=-1) - m[1:]  # (B, C, 2)
        # prior of each grid cell (log b, log c) ...
        self.log_prior = -0.5 * np.einsum("bci,ij,bcj->bc", grid, Sbc_inv, grid)
        # ... and Gaussian conditioning: prior of a given the cell
        self.a_mu0 = m[0] + grid @ (Sbc_inv @ S[1:, 0])                         # (B, C)
        self.a_var0 = S[0, 0] - S[0, 1:] @ Sbc_inv @ S[1:, 0]
        return self

    def fleet_curve(self, i, cycles):
        a, b, c = self.fleet[i]
        return a + b * np.exp(c * (np.asarray(cycles) / TIME_SCALE - EXP_T_REF))

    def prefix_stats(self, cycles, hi):
        F = self.basis(cycles)
        return {"n": np.arange(1, len(hi) + 1), "Sf": np.cumsum(F, axis=0), "Sff": np.cumsum(F * F, axis=0),
                "Sfy": np.cumsum(F * hi[:, None], axis=0), "Sy": np.cumsum(hi), "Syy": np.cumsum(hi**2)}

    def posterior(self, s, k):
        """Posterior weight of every (log b, c) cell, and posterior mean/var of a in that cell."""
        b = np.exp(EXP_LOG_B)[:, None]                                  # (B, 1)
        n = s["n"][k]
        # residuals r = y - b f(t): their sum and sum of squares, for every cell at once
        Sr = s["Sy"][k] - b * s["Sf"][k]
        Srr = s["Syy"][k] - 2 * b * s["Sfy"][k] + b**2 * s["Sff"][k]
        prec_n = 1 / self.a_var0 + n / self.sigma2
        h = self.a_mu0 / self.a_var0 + Sr / self.sigma2
        # log marginal likelihood up to a constant shared by all cells
        log_ml = -0.5 * (Srr / self.sigma2 + self.a_mu0**2 / self.a_var0 - h**2 / prec_n)
        log_w = self.log_prior + log_ml
        w = np.exp(log_w - log_w.max())
        return w / w.sum(), h / prec_n, 1 / prec_n

    def sample_params(self, stats, k, n, rng):
        w, a_mu, a_var = self.posterior(stats, k)
        cell = rng.choice(w.size, size=n, p=w.ravel())
        ib, ic = np.unravel_index(cell, w.shape)
        a = a_mu[ib, ic] + np.sqrt(a_var) * rng.standard_normal(n)
        # jitter within the grid cell so samples are not stuck on grid values
        log_b = EXP_LOG_B[ib] + rng.uniform(-0.5, 0.5, n) * np.diff(EXP_LOG_B)[0]
        log_c = np.log(EXP_RATES[ic]) + rng.uniform(-0.5, 0.5, n) * np.diff(np.log(EXP_RATES))[0]
        return np.column_stack([a, np.exp(log_b), np.exp(log_c)])

    def curves(self, params, cycles):
        t = np.asarray(cycles, dtype=float) / TIME_SCALE
        return params[:, [0]] + params[:, [1]] * np.exp(np.outer(params[:, 2], t - EXP_T_REF))

    def crossing_rul(self, p, D, cycle_now):
        a, b, c = p.T
        t0 = cycle_now / TIME_SCALE
        already = a + b * np.exp(c * (t0 - EXP_T_REF)) >= D
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cross = EXP_T_REF + np.log((D - a) / b) / c
        rul = np.where((b > 0) & (D > a), (t_cross - t0) * TIME_SCALE, np.inf)  # b <= 0: never rises
        rul[already] = 0.0
        return np.maximum(rul, 0.0)


models = [QuadraticModel().fit(train), ExponentialModel().fit(train)]
for m in models:
    m.describe()
exp_model = models[1]
print(f"[exponential] fitted rates c: median {np.median(exp_model.fleet[:, 2]):.2f}, "
      f"range {exp_model.fleet[:, 2].min():.2f}-{exp_model.fleet[:, 2].max():.2f}, "
      f"{exp_model.edge_share:.0%} of engines at the grid edge")


def fleet_baseline(cycle):
    """No-sensor baseline: median remaining life of training engines that lived past this flight."""
    rem = train_lives[train_lives > cycle] - cycle
    return np.median(rem) if len(rem) else 0.0


# %% Validation: every flight from 20 on, as if the engine stopped there
val_pred = val.loc[val["Cycle"] >= EVAL_START, ["EngineID", "Cycle", "RUL"]].reset_index(drop=True)
val_pred["pred_baseline"] = val_pred["Cycle"].map(fleet_baseline)
for m in models:
    rng = np.random.default_rng(SEED)
    parts = []
    for eid, g in val.groupby("EngineID"):
        p = m.predict_engine(g, g["Cycle"][g["Cycle"] >= EVAL_START].to_numpy(), rng)
        p.insert(0, "EngineID", eid)
        parts.append(p)
    p = pd.concat(parts).rename(columns=lambda c: c if c in ("EngineID", "Cycle") else f"{c}_{m.name}")
    val_pred = val_pred.merge(p, on=["EngineID", "Cycle"])
val_pred.to_csv(OUT / "val_predictions.csv", index=False)


def metrics(true, pred):
    d = np.asarray(pred) - np.asarray(true)  # > 0 means late (predicted too much life)
    phm = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return {"MAE": np.mean(np.abs(d)), "RMSE": np.sqrt(np.mean(d**2)), "bias": d.mean(), "PHM_mean": phm.mean()}


names = ["baseline"] + [m.name for m in models]
rows = {}
for subset, df in [("all rows", val_pred), ("RUL<=125", val_pred[val_pred.RUL <= 125])]:
    for nm in names:
        rows[(subset, nm)] = metrics(df.RUL, df[f"pred_{nm}"])
report = pd.DataFrame(rows).T
for m in models:
    report.loc[("all rows", m.name), "coverage90"] = (
        (val_pred.RUL >= val_pred[f"lo90_{m.name}"]) & (val_pred.RUL <= val_pred[f"hi90_{m.name}"])).mean()

bins = pd.cut(val_pred.RUL, [-1, 25, 50, 100, 150, 200, 400])
by_bin = pd.concat({nm: (val_pred[f"pred_{nm}"] - val_pred.RUL).groupby(bins, observed=True).agg(
    RMSE=lambda e: np.sqrt((e**2).mean()), bias="mean") for nm in names}, axis=1)

pd.set_option("display.width", 160)
print(f"\nValidation, {len(val_pred)} rows from {val_pred.EngineID.nunique()} engines")
print(report.round(3).to_string())
print("\nError by true RUL (bias > 0 = predicted too much life):")
print(by_bin.round(1).to_string())
report.to_csv(OUT / "val_metrics.csv")


# %% Test: predict at each test engine's last observed flight
for m in models:
    rng = np.random.default_rng(SEED)
    parts = []
    for eid, g in test.groupby("EngineID"):
        p = m.predict_engine(g, [g["Cycle"].max()], rng)
        p.insert(0, "EngineID", eid)
        parts.append(p)
    tp = pd.concat(parts, ignore_index=True).rename(columns={"Cycle": "FlightsObserved", "pred": "RUL_pred",
                                                             "lo90": "RUL_lo90", "hi90": "RUL_hi90"})
    cols = ["RUL_pred", "RUL_lo90", "RUL_hi90"]
    tp[cols] = tp[cols].round().astype(int)
    tp.to_csv(OUT / f"test_predictions_{m.name}.csv", index=False)
    print(f"\n[{m.name}] test predictions -> {OUT / f'test_predictions_{m.name}.csv'}")
    print(tp[cols].describe().loc[["mean", "min", "50%", "max"]].round(1).to_string())


# %% Figures
COLORS = {"quadratic": "tab:blue", "exponential": "tab:orange", "baseline": "0.5"}


def band(m, cycles, hi, grid, n=400, seed=0):
    stats = m.prefix_stats(cycles, hi)
    params = m.sample_params(stats, len(cycles) - 1, n, np.random.default_rng(seed))
    return np.percentile(m.curves(params, grid), [5, 50, 95], axis=0), stats


# 1. One training engine with both fits, then the two fleets of fitted curves (the priors)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
eng = 4
g0 = train[train.EngineID == train.EngineID.unique()[eng]]
grid0 = g0.Cycle.to_numpy()
axes[0].scatter(g0.Cycle, g0.HI, s=6, color="k", alpha=0.5, label="raw HI")
for m in models:
    axes[0].plot(grid0, m.fleet_curve(eng, grid0), color=COLORS[m.name], lw=2, label=f"{m.name} fit")
axes[0].set(title=f"Training engine {g0.EngineID.iloc[0]}: two curve shapes", xlabel="Flight", ylabel="HI (higher = more degraded)")
axes[0].legend(loc="upper left")
for ax, m in zip(axes[1:], models):
    for i, life in enumerate(train_lives[:60]):
        grid = np.arange(1, life + 1)
        ax.plot(grid, m.fleet_curve(i, grid), color=COLORS[m.name], alpha=0.3, lw=0.8)
    ax.axhspan(m.thr_mu - m.thr_sd, m.thr_mu + m.thr_sd, color="tab:red", alpha=0.15)
    ax.axhline(m.thr_mu, color="tab:red", lw=1)
    ax.set(title=f"Prior: {m.name} fits of 60 engines", xlabel="Flight")
fig.tight_layout()
fig.savefig(OUT / "fig1_prior_curves.png", dpi=150)

# 2. Posterior updating for one validation engine, both models
eid = val_pred.EngineID.unique()[0]
g = val[val.EngineID == eid]
life = g.Cycle.max()
grid = np.arange(1, 451)


def plot_update(model_list, path, figsize):
    fig, axes = plt.subplots(2 * len(model_list), 3, figsize=figsize, height_ratios=[3, 1.3] * len(model_list))
    for row, m in enumerate(model_list):
        for k, frac in enumerate([0.25, 0.5, 0.75]):
            n = max(EVAL_START, int(frac * life))
            obs = g[g.Cycle <= n]
            (lo, mid, hi), stats = band(m, obs.Cycle.to_numpy(), obs.HI.to_numpy(), grid)
            ax = axes[2 * row, k]
            ax.scatter(g.Cycle, g.HI, s=4, color="0.75", label="future (unseen)")
            ax.scatter(obs.Cycle, obs.HI, s=5, color="k", label="observed")
            ax.fill_between(grid, lo, hi, color=COLORS[m.name], alpha=0.25, label="posterior 90% band")
            ax.plot(grid, mid, color=COLORS[m.name])
            ax.axhline(m.thr_mu, color="tab:red", lw=1, label="threshold")
            ax.axvline(life, color="tab:green", ls="--", label="true end of life")
            ax.set(title=f"{m.name}: engine {eid}, observed {n}/{life} flights", xlim=(0, 450),
                   ylim=(g.HI.min() - 1, m.thr_mu + 4), xlabel="Flight")
            if k == 0:
                ax.set_ylabel("HI")
                ax.legend(fontsize=7, loc="upper left")
            r = m.rul_samples(stats, n - 1, n, np.random.default_rng(1))
            hax = axes[2 * row + 1, k]
            hax.hist(r, bins=40, color=COLORS[m.name], alpha=0.7)
            hax.axvline(life - n, color="tab:green", ls="--", label=f"true RUL {life - n}")
            hax.axvline(np.median(r), color="k", label=f"predicted {np.median(r):.0f}")
            hax.set(xlabel="Predicted RUL (flights)")
            hax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


plot_update(models, OUT / "fig2_posterior_update.png", (13, 12))
plot_update([exp_model], OUT / "fig2_posterior_update_exponential.png", (13, 6.2))  # slide-sized

# 3. Validation comparison: error vs true RUL, and RMSE by RUL range
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for nm in names:
    e = (val_pred[f"pred_{nm}"] - val_pred.RUL).groupby(val_pred.RUL)
    q = e.quantile([0.1, 0.5, 0.9]).unstack().rolling(5, center=True, min_periods=1).mean()
    if nm != "baseline":
        axes[0].fill_between(q.index, q[0.1], q[0.9], color=COLORS[nm], alpha=0.15)
    axes[0].plot(q.index, q[0.5], color=COLORS[nm], label=f"{nm} (median, 10-90% band)" if nm != "baseline" else "fleet baseline (median)")
axes[0].axhline(0, color="k", lw=1)
axes[0].set(xlabel="True RUL", ylabel="Predicted - true", title="Error vs true RUL (> 0 = too optimistic)", xlim=(250, 0), ylim=(-150, 80))
axes[0].legend(fontsize=8)
rm = by_bin.xs("RMSE", axis=1, level=1)
x = np.arange(len(rm))
for i, nm in enumerate(names):
    axes[1].bar(x + (i - 1) * 0.27, rm[nm], width=0.27, color=COLORS[nm], label=nm)
axes[1].set_xticks(x, [str(b) for b in rm.index])
axes[1].set(xlabel="True RUL range", ylabel="RMSE (flights)", title="Validation RMSE by RUL range")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "fig3_validation.png", dpi=150)

# 4. Test predictions: the two models side by side
tq = pd.read_csv(OUT / "test_predictions_quadratic.csv")
te = pd.read_csv(OUT / "test_predictions_exponential.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), width_ratios=[1, 2])
axes[0].scatter(tq.RUL_pred, te.RUL_pred, s=12)
lim = max(tq.RUL_pred.max(), te.RUL_pred.max()) + 10
axes[0].plot([0, lim], [0, lim], "k--", lw=1)
axes[0].set(xlabel="Quadratic prediction", ylabel="Exponential prediction", title="Test engines: model agreement")
order = te.RUL_pred.sort_values().index
for df, nm, dx in [(tq, "quadratic", -0.2), (te, "exponential", 0.2)]:
    s = df.loc[order].reset_index(drop=True)
    axes[1].errorbar(s.index + dx, s.RUL_pred, yerr=[s.RUL_pred - s.RUL_lo90, s.RUL_hi90 - s.RUL_pred],
                     fmt="o", ms=2.5, elinewidth=0.6, color=COLORS[nm], label=nm)
axes[1].set(xlabel="Test engines (sorted by exponential prediction)", ylabel="Predicted RUL (90% interval)", ylim=(0, 400))
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "fig4_test_predictions.png", dpi=150)

if "ipykernel" in sys.modules:
    plt.show()
print(f"Figures saved in {OUT}")
