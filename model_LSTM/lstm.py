"""Elementary LSTM prototype for RUL prediction.

Reads the per-flight feature tables written by features.ipynb, turns each
engine's history into fixed-length windows, and trains a small LSTM to predict
the RUL at the last flight of each window.

Optional ridge (L2) and lasso (L1) penalties on every weight matrix of the
network (LSTM input and recurrent weights, and the linear head; not biases).

Run (from anywhere):  python model_LSTM/lstm.py                # no penalty
                      python model_LSTM/lstm.py --l2 1e-5      # ridge
                      python model_LSTM/lstm.py --l1 1e-6      # lasso
Predictions are written to model_LSTM/lstm_test_predictions.csv.

tune_lstm.py imports the functions below to search hyperparameters.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent      # model_LSTM/
FEATURES_DIR = HERE.parent / "Features"     # written by features.ipynb

FEATURES = ["Ps30", "T50", "BPR", "htBleed", "T24", "T30",
            "health", "health_delta", "health_slope", "Cycle"]
RUL_CAP = 125      # piecewise-linear target: RUL is flat early in life

DEFAULTS = dict(
    window=40,     # flights per input sequence
    hidden=99,
    layers=2,
    dropout=0.06574206335874716,   # applied to the last hidden state, before the head
    lr=0.0006506298616959532,
    batch=128,
    epochs=30,
    l1=1.312723113278327e-08,        # lasso strength
    l2=6.1731590126622404e-06,        # ridge strength
    seed=0,
)


# ---------------------------------------------------------------- data
def load_data():
    """Train, validation and test feature tables, standardized on train."""
    train = pd.read_csv(FEATURES_DIR / "features_train.csv")
    val = pd.read_csv(FEATURES_DIR / "features_val.csv")
    test = pd.read_csv(FEATURES_DIR / "features_test.csv")
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    for df in (train, val, test):
        df[FEATURES] = (df[FEATURES] - mu) / sd
    return train, val, test


def make_windows(df, window, with_target=True, last_only=False):
    """One window ending at every flight of every engine.

    Each engine is left-padded by repeating its first row, so every flight
    (even on the shortest test engine) ends a full-length window.
    """
    X, y, ids = [], [], []
    for eid, g in df.groupby("EngineID", sort=True):
        g = g.sort_values("Cycle")
        feats = g[FEATURES].to_numpy(np.float32)
        feats = np.vstack([np.repeat(feats[:1], window - 1, axis=0), feats])
        rows = [len(g) - 1] if last_only else range(len(g))
        for r in rows:                      # r indexes the engine's real rows
            X.append(feats[r:r + window])
            ids.append(eid)
        if with_target:
            rul = np.minimum(g["RUL"].to_numpy(np.float32), RUL_CAP)
            y.extend(rul[list(rows)])
    X = np.stack(X)
    y = np.array(y, np.float32) if with_target else None
    return X, y, np.array(ids)


# ---------------------------------------------------------------- model
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (batch, window, n_features)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)  # last step only


def penalty(model, l1, l2):
    """Lasso + ridge terms over weight matrices only (biases are not shrunk)."""
    if l1 == 0 and l2 == 0:
        return 0.0
    weights = [p for n, p in model.named_parameters() if "weight" in n]
    return (l1 * sum(w.abs().sum() for w in weights)
            + l2 * sum(w.pow(2).sum() for w in weights))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy() * RUL_CAP


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def train_model(cfg, X_tr, y_tr, X_va, y_va, on_epoch=None, verbose=False):
    """Train one LSTM. Returns the model and its validation RMSE per epoch.

    on_epoch(epoch, val_rmse) is called after every epoch; tune_lstm.py uses
    it to report progress to Optuna and stop unpromising trials early.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr / RUL_CAP)),
        batch_size=cfg["batch"], shuffle=True)
    model = LSTMRegressor(X_tr.shape[2], cfg["hidden"], cfg["layers"],
                          cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            mse = loss_fn(model(xb), yb)
            loss = mse + penalty(model, cfg["l1"], cfg["l2"])
            loss.backward()
            optimizer.step()
            total += mse.item() * len(xb)   # report the fit, not the penalty
        val_rmse = rmse(predict(model, X_va), y_va)
        history.append(val_rmse)
        if verbose and (epoch == 1 or epoch % 5 == 0):
            train_rmse = np.sqrt(total / len(X_tr)) * RUL_CAP
            print(f"epoch {epoch:3d}  train RMSE {train_rmse:6.2f}  "
                  f"val RMSE {val_rmse:6.2f}")
        if on_epoch is not None:
            on_epoch(epoch, val_rmse)
    return model, history


# ---------------------------------------------------------------- script
def main():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    cfg = vars(parser.parse_args())

    train, val, test = load_data()
    w = cfg["window"]
    X_tr, y_tr, _ = make_windows(train, w)
    X_va, y_va, _ = make_windows(val, w)
    X_va_last, y_va_last, _ = make_windows(val, w, last_only=True)
    X_te, _, te_ids = make_windows(test, w, with_target=False, last_only=True)
    print(f"train windows {X_tr.shape}, val windows {X_va.shape}, "
          f"test {X_te.shape}")

    model, _ = train_model(cfg, X_tr, y_tr, X_va, y_va, verbose=True)

    print(f"\nL1 = {cfg['l1']:g}, L2 = {cfg['l2']:g}")
    # Size of each feature's input weights into the LSTM: lasso should push
    # the least useful features towards zero.
    w_in = model.lstm.weight_ih_l0.detach().abs().sum(dim=0).numpy()
    print("input weight L1 norm per feature:")
    for f, v in sorted(zip(FEATURES, w_in), key=lambda t: -t[1]):
        print(f"  {f:13s} {v:6.2f}")
    print(f"\nval RMSE, all windows      : "
          f"{rmse(predict(model, X_va), y_va):.2f}")
    print(f"val RMSE, last flight only : "
          f"{rmse(predict(model, X_va_last), y_va_last):.2f}"
          "  (true RUL ~0 here)")

    pd.DataFrame({"EngineID": te_ids,
                  "RUL_pred": predict(model, X_te).round(1)}) \
        .to_csv(HERE / "lstm_test_predictions.csv", index=False)
    print("test predictions saved to", HERE / "lstm_test_predictions.csv")


if __name__ == "__main__":
    main()
