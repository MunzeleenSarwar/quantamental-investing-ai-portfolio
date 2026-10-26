"""Quantamental Investing and AI - LSTM strategy (redo of the V1 LSTM, with V2 fixes).

V2 fixes applied:
- Scaler fit on the train window ONLY, then applied to validation and test.
- Early stopping on a validation tail of the train window (V1 ran 50 epochs
  and overfit; here training stops when validation loss stops improving).
- Daily data (about 2,500 points) instead of weekly (about 470), the sample
  size problem flagged in the V2 plan.
- Baselines reported alongside: naive train-mean prediction and Ridge
  regression on the same 60-day windows.
- Trading uses Friday predictions for the following week only, so no
  future information enters any decision.
Architecture (kept from V1 for comparability): one LSTM layer, 64 units,
Dropout 0.2, 60-day window, Dense output of 10 next-day returns.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from engine import SHORT, load_weekly, cap_weights, backtest, metrics

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

BASE = "."
TICKERS = [t + ".TO" for t in SHORT]
# ---- Daily data ----
daily = yf.download(TICKERS, start="2016-06-01", end="2026-11-01",
                    interval="1d", auto_adjust=True, progress=False)["Close"]
daily.columns = SHORT
daily = daily.dropna(how="all")
daily.to_csv("data/daily_prices.csv")
rets_d = daily.pct_change().dropna()
rets_d = rets_d.clip(-0.25, 0.25)  # winsorise extreme single-day jumps
print(f"daily rows: {len(rets_d)} from {rets_d.index[0].date()} to {rets_d.index[-1].date()}")

WINDOW = 60
vals = rets_d.values
dates = rets_d.index
# sequences: X = 60 days, y = next day
X_all, y_all, y_dates = [], [], []
for i in range(WINDOW, len(vals)):
    X_all.append(vals[i - WINDOW:i])
    y_all.append(vals[i])
    y_dates.append(dates[i])
X_all = np.array(X_all)
y_all = np.array(y_all)
y_dates = pd.DatetimeIndex(y_dates)

train_mask = y_dates < "2022-01-01"
test_mask = y_dates >= "2022-01-01"
X_tr, y_tr = X_all[train_mask], y_all[train_mask]
X_te, y_te = X_all[test_mask], y_all[test_mask]
te_dates = y_dates[test_mask]
# validation = last 20% of train sequences (still before test)
n_val = int(0.2 * len(X_tr))
X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]

scaler = StandardScaler().fit(y_tr)  # fit on train returns only
X_fit_s = scaler.transform(X_fit.reshape(-1, 10)).reshape(X_fit.shape)
X_val_s = scaler.transform(X_val.reshape(-1, 10)).reshape(X_val.shape)
X_te_s = scaler.transform(X_te.reshape(-1, 10)).reshape(X_te.shape)
y_fit_s = scaler.transform(y_fit)
y_val_s = scaler.transform(y_val)

import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(WINDOW, 10)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True, verbose=1)
hist = model.fit(X_fit_s, y_fit_s, validation_data=(X_val_s, y_val_s),
                 epochs=50, batch_size=128, callbacks=[es], verbose=0)
n_epochs = len(hist.history["loss"])
print(f"LSTM trained: {n_epochs} epochs (early stopping), "
      f"final train loss={hist.history['loss'][-1]:.5f}, "
      f"val loss={hist.history['val_loss'][-1]:.5f}")

pred_te_s = model.predict(X_te_s, verbose=0)
pred_te = scaler.inverse_transform(pred_te_s)
pred_tr_s = model.predict(X_val_s, verbose=0)
pred_val = scaler.inverse_transform(pred_tr_s)

def mse(a, b):
    return float(np.mean((a - b) ** 2))

# ---- Baselines (same windows) ----
naive_pred = np.tile(y_tr.mean(axis=0), (len(y_te), 1))
ridge = Ridge(alpha=10.0)
X_fit_flat = X_fit.reshape(len(X_fit), -1)
X_te_flat = X_te.reshape(len(X_te), -1)
ridge_preds = []
for j in range(10):
    ridge.fit(X_fit_flat, y_fit[:, j])
    ridge_preds.append(ridge.predict(X_te_flat))
ridge_pred = np.array(ridge_preds).T

mse_table = {
    "LSTM test MSE": mse(pred_te, y_te),
    "Ridge test MSE": mse(ridge_pred, y_te),
    "Naive mean test MSE": mse(naive_pred, y_te),
    "LSTM val MSE": mse(pred_val, y_val),
}
print(mse_table)

# ---- Weekly trading: Friday prediction -> next week weights ----
_, rets_w, bench_w, _ = load_weekly(BASE)
test_w = rets_w[rets_w.index >= "2022-01-01"]
bench_test = bench_w[bench_w.index >= "2022-01-01"]
test_w = test_w.reindex(bench_test.index).dropna(how="all")
static_w = pd.read_csv("data/static_weights.csv", index_col=0)
w_minvar = static_w.loc["Min Variance"].values

pred_df = pd.DataFrame(pred_te, index=te_dates, columns=SHORT)
# signal date for a week = prediction date of the last trading day before the week
# (prediction made on day t forecasts day t+1; use the latest prediction dated
# before the week starts, i.e. Friday's model output for the week ahead)
w_path = pd.DataFrame(index=test_w.index, columns=SHORT, dtype=float)
for wk in test_w.index:
    past = pred_df[pred_df.index < wk]
    if len(past) == 0:
        w_path.loc[wk] = w_minvar
        continue
    sig = past.iloc[-1].values
    pos = np.clip(sig, 0, None)
    if pos.sum() <= 0:
        w_path.loc[wk] = w_minvar
    else:
        w_path.loc[wk] = cap_weights(pos / pos.sum())
w_path = w_path.astype(float)
bt = backtest(test_w, w_path)
met = metrics(bt["ret"], bench_test.reindex(test_w.index))
met["avg_weekly_turnover"] = bt["turnover"].mean()
met["annual_turnover"] = bt["turnover"].mean() * 52
met["total_cost_drag"] = bt["cost"].sum()
print(f"LSTM strategy: ret={met['ann_return']*100:.1f}% vol={met['ann_vol']*100:.1f}% "
      f"Sharpe={met['sharpe']:.2f} maxDD={met['max_drawdown']*100:.1f}%")

# per-year test returns (walk-forward style evaluation blocks)
bt_df = bt.copy()
bt_df["year"] = bt_df.index.year
yearly = {}
for yr, grp in bt_df.groupby("year"):
    yearly[str(yr)] = {"ann_return": float((1 + grp["ret"]).prod() ** (52 / len(grp)) - 1),
                       "weeks": int(len(grp))}

growth = (1 + bt["ret"]).cumprod() * 100
growth.to_csv("data/lstm_growth.csv", header=["LSTM"])
w_path.to_csv("data/lstm_weights.csv")
with open("data/lstm_results.json", "w") as f:
    json.dump({"metrics": {k: float(v) for k, v in met.items()},
               "mse": mse_table,
               "epochs": int(n_epochs),
               "train_loss": float(hist.history["loss"][-1]),
               "val_loss": float(hist.history["val_loss"][-1]),
               "history": {"loss": [float(x) for x in hist.history["loss"]],
                           "val_loss": [float(x) for x in hist.history["val_loss"]]},
               "yearly": yearly}, f, indent=2)
print("saved data/lstm_results.json, data/lstm_growth.csv")
