"""Quantamental Investing and AI - static models: Equal Weight, Max Sharpe (Monte Carlo + optimiser),
CAPM inverse-beta, Minimum Variance, Risk Parity, Fama-French 5 factor.

All weights are estimated on the train window only (Jun 2016 - Dec 2021) and
held through the test window (Jan 2022 - Jun 2026) with weekly rebalancing
back to target. Caps: 20% per name, 40% per sector, long-only.
"""
import json, sys, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from engine import (TICKERS, SHORT, NAMES, SECTORS, RF_WEEKLY, CAP,
                    load_weekly, cap_weights, backtest, metrics)
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
BASE = "."
px, rets, bench, vix = load_weekly(BASE)
train = rets[rets.index < "2022-01-01"]
test = rets[rets.index >= "2022-01-01"]
bench_train = bench[bench.index < "2022-01-01"]
bench_test = bench[bench.index >= "2022-01-01"]
test = test.reindex(bench_test.index).dropna(how="all")
print(f"train weeks: {len(train)}, test weeks: {len(test)}")
print(f"train: {train.index[0].date()} to {train.index[-1].date()}")
print(f"test:  {test.index[0].date()} to {test.index[-1].date()}")

mu = train.mean().values * 52  # annualised mean returns
mu_shrunk = 0.5 * mu + 0.5 * mu.mean()  # shrink noisy means toward grand mean
lw = LedoitWolf().fit(train.values)
cov_w = lw.covariance_ * 52  # annualised shrunk covariance
sd = np.sqrt(np.diag(cov_w))

# ---- CAPM betas (OLS, excess returns both sides, train window) ----
bench_tr = bench_train.reindex(train.index).dropna()
tr = train.reindex(bench_tr.index)
excess = tr.values - RF_WEEKLY
bench_ex = bench_tr.values - RF_WEEKLY
betas = {}
for i, t in enumerate(SHORT):
    x = np.vstack([np.ones_like(bench_ex), bench_ex]).T
    coef, *_ = np.linalg.lstsq(x, excess[:, i], rcond=None)
    betas[t] = coef[1]
print("CAPM betas (train):")
for t in SHORT:
    print(f"  {t:5s} beta={betas[t]:.2f}")

# ---- Optimiser helpers ----
cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
bnds = [(0, CAP)] * len(SHORT)
# sector constraints: sum of members <= 0.40
from engine import SECTOR_MEMBERS
idx = {t: i for i, t in enumerate(SHORT)}
for sec, members in SECTOR_MEMBERS.items():
    ii = [idx[m] for m in members]
    cons.append({"type": "ineq",
                 "fun": (lambda ii: (lambda w: 0.40 - w[ii].sum()))(ii)})


def max_sharpe(mu_, cov_):
    def neg_sharpe(w):
        ret = w @ mu_
        vol = np.sqrt(w @ cov_ @ w)
        return -(ret - 0.02) / vol if vol > 0 else 0.0
    res = minimize(neg_sharpe, np.ones(len(SHORT)) / len(SHORT),
                   method="SLSQP", bounds=bnds, constraints=cons)
    return cap_weights(res.x)


def min_var(cov_):
    res = minimize(lambda w: w @ cov_ @ w, np.ones(len(SHORT)) / len(SHORT),
                   method="SLSQP", bounds=bnds, constraints=cons)
    return cap_weights(res.x)


def risk_parity(cov_):
    def obj(w):
        port_vol = np.sqrt(w @ cov_ @ w)
        rc = w * (cov_ @ w) / port_vol if port_vol > 0 else w * 0
        target = port_vol / len(w)
        return np.sum((rc - target) ** 2)
    res = minimize(obj, np.ones(len(SHORT)) / len(SHORT),
                   method="SLSQP", bounds=bnds, constraints=cons)
    return cap_weights(res.x)


weights = {}
weights["Equal Weight"] = np.ones(len(SHORT)) / len(SHORT)
weights["Max Sharpe"] = max_sharpe(mu_shrunk, cov_w)
weights["Min Variance"] = min_var(cov_w)
weights["Risk Parity"] = risk_parity(cov_w)
inv_beta = np.array([1.0 / max(betas[t], 0.05) for t in SHORT])
weights["CAPM Inverse-Beta"] = cap_weights(inv_beta / inv_beta.sum())

# ---- Monte Carlo frontier (for the chart, and MC max-Sharpe check) ----
rng = np.random.default_rng(42)
mc_w, mc_ret, mc_vol = [], [], []
tries = 0
while len(mc_w) < 20000 and tries < 400000:
    tries += 1
    w = rng.dirichlet(np.ones(len(SHORT)))
    if np.all(w <= CAP):
        mc_w.append(w)
        mc_ret.append(w @ mu_shrunk)
        mc_vol.append(np.sqrt(w @ cov_w @ w))
mc_w = np.array(mc_w); mc_ret = np.array(mc_ret); mc_vol = np.array(mc_vol)
mc_sharpe = (mc_ret - 0.02) / mc_vol
mc_best = mc_w[np.argmax(mc_sharpe)]
print(f"Monte Carlo: {len(mc_w)} valid portfolios from {tries} draws; "
      f"best MC Sharpe={mc_sharpe.max():.2f}")

# ---- Fama-French 5 (monthly, French US factors, stated as approximation) ----
ff5_weights = None
try:
    import io, zipfile, urllib.request
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    txt = zf.read(name).decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in txt:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6 and parts[0].isdigit() and len(parts[0]) == 6:
            rows.append(parts[:6])
    fac = pd.DataFrame(rows, columns=["yyyymm", "Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    fac["date"] = pd.to_datetime(fac["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    fac = fac.set_index("date").drop(columns="yyyymm").astype(float) / 100.0
    # monthly stock returns from weekly prices
    mpx = px[SHORT].resample("ME").last()
    mret = mpx.pct_change().dropna()
    # risk-free monthly from French RF column is not in this file; use Mkt-RF relation:
    # approximate monthly RF as 0.02/12
    rf_m = 0.02 / 12
    fac_tr = fac[(fac.index >= "2016-06-01") & (fac.index < "2022-01-01")]
    common = mret.index.intersection(fac_tr.index)
    # align to month ends present in both
    mret_tr = mret.reindex(common).dropna()
    fac_tr = fac_tr.reindex(mret_tr.index)
    alphas, fbetas = {}, {}
    for t in SHORT:
        y = mret_tr[t].values - rf_m
        X = np.column_stack([np.ones(len(y))] + [fac_tr[c].values for c in
                            ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alphas[t] = coef[0]
        fbetas[t] = coef[1:]
    fmean = fac_tr[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].mean().values
    mu_ff = np.array([alphas[t] + fbetas[t] @ fmean for t in SHORT]) * 12
    cov_m = np.cov(mret_tr.values.T) * 12
    lw_m = LedoitWolf().fit(mret_tr.values)
    cov_m = lw_m.covariance_ * 12
    ff5_weights = max_sharpe(mu_ff, cov_m)
    weights["Fama-French 5"] = ff5_weights
    print("FF5 expected returns (annualised):")
    for t, v in zip(SHORT, mu_ff):
        print(f"  {t:5s} {v * 100:6.1f}%  alpha={alphas[t] * 100:5.2f}%/mo")
except Exception as e:
    print(f"FF5 skipped (factor download failed): {e}")

# ---- Backtest every static strategy on the test window ----
results = {}
wrows = {}
for name, w in weights.items():
    wdf = pd.DataFrame(np.tile(w, (len(test), 1)), index=test.index, columns=SHORT)
    bt = backtest(test, wdf)
    met = metrics(bt["ret"], bench_test.reindex(test.index))
    met["avg_weekly_turnover"] = bt["turnover"].mean()
    met["annual_turnover"] = bt["turnover"].mean() * 52
    met["total_cost_drag"] = bt["cost"].sum()
    results[name] = met
    wrows[name] = w
    print(f"{name:18s} ret={met['ann_return']*100:6.1f}% vol={met['ann_vol']*100:5.1f}% "
          f"Sharpe={met['sharpe']:5.2f} maxDD={met['max_drawdown']*100:6.1f}% "
          f"CVaR95w={met['cvar95_weekly']*100:5.2f}% turnover={met['annual_turnover']*100:5.0f}%/yr")

bench_met = metrics(bench_test.reindex(test.index), bench_test.reindex(test.index))
results["S&P/TSX Benchmark"] = bench_met
print(f"{'S&P/TSX Benchmark':18s} ret={bench_met['ann_return']*100:6.1f}% vol={bench_met['ann_vol']*100:5.1f}% "
      f"Sharpe={bench_met['sharpe']:5.2f} maxDD={bench_met['max_drawdown']*100:6.1f}%")

# ---- Save ----
wdf_out = pd.DataFrame(wrows, index=SHORT).T
wdf_out.to_csv("data/static_weights.csv")
with open("data/static_results.json", "w") as f:
    json.dump({"results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
               "betas": betas,
               "mc_best_sharpe": float(mc_sharpe.max()),
               "n_mc": int(len(mc_w))}, f, indent=2)
np.savez("data/mc_frontier.npz", w=mc_w, ret=mc_ret, vol=mc_vol)
# cumulative growth series for charts
growth = {}
for name, w in weights.items():
    wdf = pd.DataFrame(np.tile(w, (len(test), 1)), index=test.index, columns=SHORT)
    bt = backtest(test, wdf)
    growth[name] = (1 + bt["ret"]).cumprod() * 100
growth["S&P/TSX Benchmark"] = (1 + bench_test.reindex(test.index)).cumprod() * 100
pd.DataFrame(growth).to_csv("data/static_growth.csv")
print("saved data/static_weights.csv, data/static_results.json, data/static_growth.csv")
