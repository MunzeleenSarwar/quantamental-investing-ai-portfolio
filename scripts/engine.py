"""Quantamental Investing and AI - shared backtest engine and helpers.

Design (from the V2 redo plan):
- Weekly TSX prices in CAD, Jun 2016 - Oct 2026.
- Estimation window: Jun 2016 - Dec 2021. Test window: Jan 2022 - Oct 2026.
  Headline results are test-window only, for every strategy.
- Long-only, 0-20% per stock, 40% per sector, weights sum to 100%.
- Transaction cost: 10 bps on traded notional (sum of absolute weight changes).
- Risk-free rate: 2.0% annual, stated as an assumption.
- Rebalancing: weekly back to target weights for static strategies.
"""
import numpy as np
import pandas as pd

TICKERS = ["CNQ.TO", "ENB.TO", "CCO.TO", "TD.TO", "BN.TO",
            "SHOP.TO", "TIH.TO", "ABX.TO", "L.TO", "CAE.TO"]
SHORT = ["CNQ", "ENB", "CCO", "TD", "BN", "SHOP", "TIH", "ABX", "L", "CAE"]
NAMES = {"CNQ": "Canadian Natural Resources", "ENB": "Enbridge", "CCO": "Cameco",
         "TD": "TD Bank", "BN": "Brookfield Corp", "SHOP": "Shopify",
         "TIH": "Toromont", "ABX": "Barrick Mining", "L": "Loblaw", "CAE": "CAE"}
SECTORS = {"CNQ": "Energy", "ENB": "Energy", "CCO": "Energy",
           "TD": "Financials", "BN": "Financials", "SHOP": "Technology",
           "TIH": "Industrials", "CAE": "Industrials",
           "ABX": "Materials", "L": "Consumer Staples"}
SECTOR_MEMBERS = {}
for _t, _s in SECTORS.items():
    SECTOR_MEMBERS.setdefault(_s, []).append(_t)

RF_ANNUAL = 0.02
RF_WEEKLY = RF_ANNUAL / 52.0
COST_BPS = 0.0010  # 10 bps on traded notional
CAP = 0.20
SECTOR_CAP = 0.40
TEST_START = "2022-01-01"


def load_weekly(base):
    px = pd.read_csv(f"{base}/data/weekly_prices.csv", index_col=0, parse_dates=True)
    px = px.rename(columns={t: s for t, s in zip(TICKERS, SHORT)})
    rets = px[SHORT].pct_change().dropna(how="all")
    bench = px["^GSPTSE"].pct_change().dropna()
    vix = px["^VIX"]
    return px, rets, bench, vix


def cap_weights(w, cap=CAP, sector_cap=SECTOR_CAP, iters=200):
    """Iteratively cap single names and sectors, renormalising to sum 1."""
    w = np.array(w, dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return np.ones_like(w) / len(w)
    w = w / w.sum()
    idx = {t: i for i, t in enumerate(SHORT)}
    for _ in range(iters):
        changed = False
        w = np.clip(w, 0, cap)
        for sec, members in SECTOR_MEMBERS.items():
            ii = [idx[m] for m in members]
            if w[ii].sum() > sector_cap:
                w[ii] = w[ii] * (sector_cap / w[ii].sum())
                changed = True
        s = w.sum()
        if s == 0:
            return np.ones_like(w) / len(w)
        w = w / s
        if not changed and np.all(w <= cap + 1e-9):
            break
    return w / w.sum()


def backtest(rets_test, weights_over_time, bench_test=None):
    """Run a weight path through the test window.

    weights_over_time: DataFrame of target weights, indexed like rets_test
    (rows may repeat the same target for many weeks).
    Returns dict with weekly net returns, turnover and cost series.
    """
    w_prev_drifted = None
    port_rets, turnovers, costs = [], [], []
    for date, row in weights_over_time.iterrows():
        target = row.values.astype(float)
        r = rets_test.loc[date].values.astype(float)
        if w_prev_drifted is None:
            turnover = 0.0  # initial purchase is the starting position, not a trade cost here
            cost = 0.0
        else:
            turnover = np.abs(target - w_prev_drifted).sum()
            cost = COST_BPS * turnover
        gross = float(target @ r)
        port_rets.append(gross - cost)
        turnovers.append(turnover)
        costs.append(cost)
        # drift weights by realised returns, then next week trades from drifted weights
        grown = target * (1.0 + r)
        w_prev_drifted = grown / grown.sum()
    out = pd.DataFrame({"ret": port_rets, "turnover": turnovers, "cost": costs},
                       index=weights_over_time.index)
    return out


def metrics(port_rets, bench_rets=None):
    """Annualised metrics from a weekly net-return series."""
    r = pd.Series(port_rets).dropna()
    n = len(r)
    ann_ret = (1 + r).prod() ** (52.0 / n) - 1 if n else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(52)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol else np.nan
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    cvar = r[r <= r.quantile(0.05)].mean()  # weekly CVaR95
    downside = r[r < 0]
    sortino = (ann_ret - RF_ANNUAL) / (downside.std(ddof=1) * np.sqrt(52)) if len(downside) > 1 else np.nan
    calmar = ann_ret / abs(max_dd) if max_dd else np.nan
    m = {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
         "max_drawdown": max_dd, "cvar95_weekly": cvar,
         "sortino": sortino, "calmar": calmar,
         "worst_week": r.min(), "total_return": cum.iloc[-1] - 1 if n else np.nan}
    if bench_rets is not None:
        b = pd.Series(bench_rets).reindex(r.index).dropna()
        rr = r.reindex(b.index)
        m["beta_vs_bench"] = (np.cov(rr, b)[0, 1] / np.var(b)) if np.var(b) else np.nan
        m["tracking_error"] = (rr - b).std(ddof=1) * np.sqrt(52)
        m["info_ratio"] = ((rr - b).mean() * 52) / m["tracking_error"] if m["tracking_error"] else np.nan
    return m


def fmt_pct(x, d=1):
    return f"{x * 100:.{d}f}%"
