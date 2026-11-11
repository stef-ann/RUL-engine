"""Bayesian hyperparameter search for the LSTM in lstm.py, using Optuna.

Optuna's TPE sampler builds a probabilistic model of which settings give low
validation RMSE and proposes the next trial from it. A median pruner stops
trials whose validation RMSE is worse than the median of earlier trials at the
same epoch, so bad settings don't run to completion.

The 52 validation engines are split in half by engine:
  - "tune" engines: score every trial (the number Optuna minimises)
  - "holdout" engines: scored once, for the best setting only, so the
    reported RMSE was not used to choose the hyperparameters.
The test engines are not touched until the best model makes its predictions.

Run (from anywhere):  python model_LSTM/tune_lstm.py                    # 30 trials
                      python model_LSTM/tune_lstm.py --trials 50 --timeout 3600
Results are stored in model_LSTM/optuna_lstm.db, so a rerun continues the same study.
"""

import argparse

import optuna
import pandas as pd

from lstm import (HERE, DEFAULTS, load_data, make_windows, predict, rmse,
                  train_model)

MAX_EPOCHS = 30

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=30)
parser.add_argument("--timeout", type=int, default=None, help="seconds")
args = parser.parse_args()

train, val, test = load_data()
val_ids = sorted(val["EngineID"].unique())
tune_ids, hold_ids = val_ids[0::2], val_ids[1::2]     # alternate engines
val_tune = val[val["EngineID"].isin(tune_ids)]
val_hold = val[val["EngineID"].isin(hold_ids)]

_windows = {}


def windows(w):
    """Windowed arrays for a given window length, built once and cached."""
    if w not in _windows:
        _windows[w] = (make_windows(train, w), make_windows(val_tune, w))
    return _windows[w]


def objective(trial):
    cfg = dict(DEFAULTS)
    cfg.update(
        window=trial.suggest_int("window", 10, 50, step=10),
        hidden=trial.suggest_int("hidden", 16, 128, log=True),
        layers=trial.suggest_int("layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch=trial.suggest_categorical("batch", [128, 256, 512]),
        l1=trial.suggest_float("l1", 1e-9, 1e-4, log=True),
        l2=trial.suggest_float("l2", 1e-9, 1e-3, log=True),
        epochs=MAX_EPOCHS,
    )
    (X_tr, y_tr, _), (X_va, y_va, _) = windows(cfg["window"])

    def on_epoch(epoch, val_rmse):
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    _, history = train_model(cfg, X_tr, y_tr, X_va, y_va, on_epoch=on_epoch)
    # Score = best epoch, and remember which epoch that was for the refit.
    best_epoch = min(range(len(history)), key=history.__getitem__) + 1
    trial.set_user_attr("best_epoch", best_epoch)
    return history[best_epoch - 1]


study = optuna.create_study(
    study_name="lstm_rul",
    storage=f"sqlite:///{(HERE / 'optuna_lstm.db').as_posix()}",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
)
study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

# ---------------------------------------------------------------- report
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
print(f"\n{len(done)} trials completed, {len(pruned)} pruned")

best = study.best_trial
print(f"best tune-engine RMSE {best.value:.2f} (trial {best.number}, "
      f"epoch {best.user_attrs['best_epoch']})")
for k, v in best.params.items():
    print(f"  {k:8s} {v:.3g}" if isinstance(v, float) else f"  {k:8s} {v}")

# Refit the best setting for its best epoch count, then score the holdout.
cfg = dict(DEFAULTS, **best.params, epochs=best.user_attrs["best_epoch"])
(X_tr, y_tr, _), (X_va, y_va, _) = windows(cfg["window"])
model, _ = train_model(cfg, X_tr, y_tr, X_va, y_va)

X_ho, y_ho, _ = make_windows(val_hold, cfg["window"])
X_ho_last, y_ho_last, _ = make_windows(val_hold, cfg["window"], last_only=True)
print(f"\nholdout RMSE, all windows      : {rmse(predict(model, X_ho), y_ho):.2f}")
print(f"holdout RMSE, last flight only : "
      f"{rmse(predict(model, X_ho_last), y_ho_last):.2f}")

X_te, _, te_ids = make_windows(test, cfg["window"], with_target=False,
                               last_only=True)
pd.DataFrame({"EngineID": te_ids, "RUL_pred": predict(model, X_te).round(1)}) \
    .to_csv(HERE / "lstm_tuned_test_predictions.csv", index=False)
print("test predictions saved to", HERE / "lstm_tuned_test_predictions.csv")
