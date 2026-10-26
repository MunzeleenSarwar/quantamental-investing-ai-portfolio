"""Quantamental Investing and AI - dynamic models: MACD gate, filtered 2-state HMM gate,
Regime-Aware CVaR (monthly re-estimated, Rockafellar-Uryasev LP).

All signals use data available at the decision date only (filtered HMM
probabilities, 2-week persistence for MACD and HMM gates). Bullish regimes
hold Max Sharpe weights; defensive regimes hold Min Variance weights.
Costs, caps and the test window are identical to the static models.
"""
import json, warnings
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from engine import SHORT, load_weekly, cap_weights, backtest, metrics
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
BASE = "."
px, rets, bench, vix = load_weekly(BASE)
train = rets[rets.index < "2022-01-01"]
test = rets[rets.index >= "2022-01-01"]
bench_train = bench[bench.index < "2022-01-01"]
bench_test = bench[bench.index >= "2022-01-01"]
test = test.reindex(bench_test.index).dropna(how="all")

static_w = pd.read_csv("data/static_weights.csv", index_col=0)
w_maxsharpe = static_w.loc["Max Sharpe"].values
w_minvar = static_w.loc["Min Variance"].values

# ---------- MACD gate on the benchmark (12/26/9, 2-week persistence) ----------
bench_px = px["^GSPTSE"]
ema12 = bench_px.ewm(span=12, adjust=False).mean()
ema26 = bench_px.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
bullish = (macd > signal).astype(int)
# decision for week t uses signals through week t-1 (no look-ahead)
bullish_lag = bullish.shift(1)
persist = (bullish_lag == 1) & (bullish_lag.shift(1) == 1)
macd_weights = pd.DataFrame(
    np.where(persist.reindex(test.index).fillna(0).values[:, None] == 1,
             w_maxsharpe, w_minvar),
    index=test.index, columns=SHORT)
bt = backtest(test, macd_weights)
met = metrics(bt["ret"], bench_test.reindex(test.index))
met["avg_weekly_turnover"] = bt["turnover"].mean()
met["annual_turnover"] = bt["turnover"].mean() * 52
met["total_cost_drag"] = bt["cost"].sum()
met["bullish_weeks_pct"] = float(persist.reindex(test.index).fillna(0).mean())
results = {"MACD Gate": met}
print(f"MACD Gate: ret={met['ann_return']*100:.1f}% vol={met['ann_vol']*100:.1f}% "
      f"Sharpe={met['sharpe']:.2f} maxDD={met['max_drawdown']*100:.1f}% "
      f"bullish={met['bullish_weeks_pct']*100:.0f}% of weeks")

# ---------- 2-state Gaussian HMM, filtered probabilities only ----------
# Features (standardised on train): benchmark weekly return, 4-week realised
# volatility of the benchmark, VIX level. GDP is dropped: V1 regime-1
# coefficients were insignificant and GDP is published with a lag.
bench_ret_full = bench.pct_change().dropna()
vol4 = bench_ret_full.rolling(4).std()
feat = pd.DataFrame({"ret": bench_ret_full,
                     "vol": vol4,
                     "vix": vix.reindex(bench_ret_full.index)}).dropna()
feat_tr = feat[feat.index < "2022-01-01"]
mu_f = feat_tr.mean().values
sd_f = feat_tr.std(ddof=1).values
X = ((feat - mu_f) / sd_f).values
Xtr = X[feat.index < "2022-01-01"]


def fit_hmm_2state(X, n_iter=200, seed=7):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # init: split by volatility feature (column 1)
    lab = (X[:, 1] > np.median(X[:, 1])).astype(int)
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.5, 0.5])
    mus = np.array([X[lab == k].mean(axis=0) for k in range(2)])
    covs = np.array([np.cov(X[lab == k].T) + 1e-3 * np.eye(d) for k in range(2)])
    for _ in range(n_iter):
        # forward
        logp = np.zeros((n, 2))
        for k in range(2):
            diff = X - mus[k]
            inv = np.linalg.inv(covs[k])
            det = np.linalg.det(covs[k])
            logp[:, k] = -0.5 * (np.sum(diff @ inv * diff, axis=1)
                                 + d * np.log(2 * np.pi) + np.log(det))
        # forward recursion in log space
        la = np.log(pi + 1e-300) + logp[0]
        alphas = [la]
        for t in range(1, n):
            m = la.max()
            la = logp[t] + m + np.log(np.exp(la - m) @ A + 1e-300)
            alphas.append(la)
        alphas = np.array(alphas)
        # backward
        lb = np.zeros(2)
        betas = [lb]
        for t in range(n - 2, -1, -1):
            m = (lb + logp[t + 1]).max()
            lb = m + np.log(np.exp(lb + logp[t + 1] - m) @ A.T + 1e-300)
            betas.append(lb)
        betas = np.array(betas[::-1])
        gamma = alphas + betas
        gamma = np.exp(gamma - gamma.max(axis=1, keepdims=True))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((n - 1, 2, 2))
        for t in range(n - 1):
            mat = (alphas[t][:, None] + np.log(A + 1e-300)
                   + logp[t + 1][None, :] + betas[t + 1][None, :])
            mat = np.exp(mat - mat.max())
            xi[t] = mat / mat.sum()
        pi = gamma[0]
        A = xi.sum(axis=0) / np.maximum(xi.sum(axis=0).sum(axis=1, keepdims=True), 1e-300)
        for k in range(2):
            g = gamma[:, k]
            mus[k] = (g[:, None] * X).sum(axis=0) / g.sum()
            diff = X - mus[k]
            covs[k] = (diff * g[:, None]).T @ diff / g.sum() + 1e-3 * np.eye(d)
    return A, pi, mus, covs


A, pi, mus, covs = fit_hmm_2state(Xtr)
# distress state = higher volatility-feature mean
distress = int(np.argmax(mus[:, 1]))
print(f"HMM states: calm vol-feature mean={mus[1-distress, 1]:.2f}, "
      f"distress vol-feature mean={mus[distress, 1]:.2f}; "
      f"persistence A={np.round(A, 3).tolist()}")


def filtered_distress_prob(X_all, A, pi, mus, covs):
    n, d = X_all.shape
    logp = np.zeros((n, 2))
    for k in range(2):
        diff = X_all - mus[k]
        inv = np.linalg.inv(covs[k])
        det = np.linalg.det(covs[k])
        logp[:, k] = -0.5 * (np.sum(diff @ inv * diff, axis=1)
                             + d * np.log(2 * np.pi) + np.log(det))
    la = np.log(pi + 1e-300) + logp[0]
    out = [np.exp(la - la.max())]
    out[0] = out[0] / out[0].sum()
    for t in range(1, n):
        la = logp[t] + np.log(np.exp(la - la.max()) @ A + 1e-300)
        p = np.exp(la - la.max())
        out.append(p / p.sum())
    return np.array(out)[:, distress]


probs = filtered_distress_prob(X, A, pi, mus, covs)
prob_s = pd.Series(probs, index=feat.index)
# decision for week t uses probs through week t-1, distress only if 2 in a row
prob_lag = prob_s.shift(1)
distress_now = (prob_lag > 0.5) & (prob_lag.shift(1) > 0.5)
hmm_weights = pd.DataFrame(
    np.where(distress_now.reindex(test.index).fillna(False).values[:, None],
             w_minvar, w_maxsharpe),
    index=test.index, columns=SHORT)
bt = backtest(test, hmm_weights)
met = metrics(bt["ret"], bench_test.reindex(test.index))
met["avg_weekly_turnover"] = bt["turnover"].mean()
met["annual_turnover"] = bt["turnover"].mean() * 52
met["total_cost_drag"] = bt["cost"].sum()
met["distress_weeks_pct"] = float(distress_now.reindex(test.index).fillna(False).mean())
results["HMM Regime Gate"] = met
print(f"HMM Gate: ret={met['ann_return']*100:.1f}% vol={met['ann_vol']*100:.1f}% "
      f"Sharpe={met['sharpe']:.2f} maxDD={met['max_drawdown']*100:.1f}% "
      f"distress={met['distress_weeks_pct']*100:.0f}% of weeks")

# ---------- Regime-Aware CVaR: monthly min-CVaR (Rockafellar-Uryasev LP) ----------
from engine import SECTOR_MEMBERS
idx = {t: i for i, t in enumerate(SHORT)}


def min_cvar_weights(scenarios, alpha_level=0.95):
    n_scen, n = scenarios.shape
    # variables: w (n), alpha (1), u (n_scen)
    c = np.concatenate([np.zeros(n), [1.0],
                        np.ones(n_scen) / ((1 - alpha_level) * n_scen)])
    # u_i >= -r_i'w - alpha  ->  r_i'w + alpha + u_i >= 0
    A_ub = []
    b_ub = []
    for i in range(n_scen):
        row = np.zeros(n + 1 + n_scen)
        row[:n] = -scenarios[i]
        row[n] = -1.0
        row[n + 1 + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = np.zeros((1, n + 1 + n_scen))
    A_eq[0, :n] = 1.0
    bounds = [(0, 0.20)] * n + [(None, None)] + [(0, None)] * n_scen
    # sector caps as additional inequalities
    for sec, members in SECTOR_MEMBERS.items():
        row = np.zeros(n + 1 + n_scen)
        for m in members:
            row[idx[m]] = 1.0
        A_ub.append(row)
        b_ub.append(0.40)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=np.array([1.0]), bounds=bounds, method="highs")
    if not res.success:
        return w_minvar
    return cap_weights(res.x[:n])


month_ends = test.resample("ME").last().index
# decision dates: last week of each month in test; weights apply next month
targets = {}
prev_w = w_minvar
all_rets = rets[rets.index < test.index[0]].copy()  # grows as test proceeds (past only)
# build month by month using only data available at each decision date
cur = test.index[0]
w_path = pd.DataFrame(index=test.index, columns=SHORT, dtype=float)
for me in month_ends:
    # decision at month end me: scenarios = last 156 weeks of returns up to me
    hist = rets[rets.index <= me]
    scen = hist.tail(156).values
    w = min_cvar_weights(scen)
    # apply to weeks after me until next month end
    nxt = month_ends[month_ends > me]
    end = nxt[0] if len(nxt) else test.index[-1]
    mask = (test.index > me) & (test.index <= end)
    w_path.loc[mask] = w
# weeks before the first month end keep MinVar
w_path = w_path.fillna(pd.Series(w_minvar, index=SHORT))
w_path = w_path.astype(float)
bt = backtest(test, w_path)
met = metrics(bt["ret"], bench_test.reindex(test.index))
met["avg_weekly_turnover"] = bt["turnover"].mean()
met["annual_turnover"] = bt["turnover"].mean() * 52
met["total_cost_drag"] = bt["cost"].sum()
results["Regime-Aware CVaR"] = met
print(f"Regime CVaR: ret={met['ann_return']*100:.1f}% vol={met['ann_vol']*100:.1f}% "
      f"Sharpe={met['sharpe']:.2f} maxDD={met['max_drawdown']*100:.1f}% "
      f"CVaR95w={met['cvar95_weekly']*100:.2f}%")

# growth series + save
growth = {}
for name, wdf in [("MACD Gate", macd_weights), ("HMM Regime Gate", hmm_weights),
                  ("Regime-Aware CVaR", w_path)]:
    bt = backtest(test, wdf)
    growth[name] = (1 + bt["ret"]).cumprod() * 100
pd.DataFrame(growth).to_csv("data/dynamic_growth.csv")
w_path.to_csv("data/cvar_weights.csv")
with open("data/dynamic_results.json", "w") as f:
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
print("saved data/dynamic_results.json, data/dynamic_growth.csv")
