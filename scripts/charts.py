"""Quantamental Investing and AI - charts from saved growth series and weights."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
NAVY = "#1f3864"
GOLD = "#bf8f00"
GREY = "#595959"

sg = pd.read_csv("data/static_growth.csv", index_col=0, parse_dates=True)
dg = pd.read_csv("data/dynamic_growth.csv", index_col=0, parse_dates=True)
bench = sg["S&P/TSX Benchmark"]

# 1. Static growth
fig, ax = plt.subplots(figsize=(10, 5.2))
for c in sg.columns:
    lw = 2.2 if c == "S&P/TSX Benchmark" else 1.4
    ls = "--" if c == "S&P/TSX Benchmark" else "-"
    ax.plot(sg.index, sg[c], label=c, lw=lw, ls=ls)
ax.set_title("Growth of $100, test window Jan 2022 - Oct 2026: static strategies")
ax.set_ylabel("$")
ax.legend(ncol=2, fontsize=8, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("charts/static_growth.png", dpi=200)
plt.close(fig)

# 2. Dynamic growth
fig, ax = plt.subplots(figsize=(10, 5.2))
for c in dg.columns:
    ax.plot(dg.index, dg[c], label=c, lw=1.4)
ax.plot(bench.index, bench, label="S&P/TSX Benchmark", lw=2.2, ls="--")
ax.set_title("Growth of $100, test window Jan 2022 - Oct 2026: dynamic strategies")
ax.set_ylabel("$")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("charts/dynamic_growth.png", dpi=200)
plt.close(fig)

# 3. Drawdown
def drawdown(growth):
    return growth / growth.cummax() - 1

fig, ax = plt.subplots(figsize=(10, 5.2))
for c in ["Min Variance", "Max Sharpe", "Regime-Aware CVaR"]:
    s = dg[c] if c in dg else sg[c]
    ax.plot(s.index, drawdown(s) * 100, label=c, lw=1.4)
ax.plot(bench.index, drawdown(bench) * 100, label="S&P/TSX Benchmark", lw=2.0, ls="--")
ax.set_title("Drawdown, test window (%)")
ax.set_ylabel("%")
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("charts/drawdown.png", dpi=200)
plt.close(fig)

# 4. Efficient frontier (train window, Monte Carlo)
z = np.load("data/mc_frontier.npz")
fig, ax = plt.subplots(figsize=(10, 5.2))
sc = ax.scatter(z["vol"] * 100, z["ret"] * 100, c=(z["ret"] - 0.02) / z["vol"],
                cmap="viridis", s=4, alpha=0.5)
ax.set_xlabel("Annualised volatility (%)")
ax.set_ylabel("Annualised return (%)")
ax.set_title("Efficient frontier, estimation window Jun 2016 - Dec 2021 (20,000 random portfolios)")
fig.colorbar(sc, label="Sharpe ratio")
fig.tight_layout()
fig.savefig("charts/frontier.png", dpi=200)
plt.close(fig)

# 5. Risk-return scatter, test window
with open("data/static_results.json") as f:
    sr = json.load(f)["results"]
with open("data/dynamic_results.json") as f:
    dr = json.load(f)
allres = {**sr, **dr}
fig, ax = plt.subplots(figsize=(10, 5.2))
for name, m in allres.items():
    if name == "S&P/TSX Benchmark":
        ax.scatter(m["ann_vol"] * 100, m["ann_return"] * 100, s=90, marker="*",
                   color=GOLD, edgecolors="black", zorder=5)
        ax.annotate(name, (m["ann_vol"] * 100, m["ann_return"] * 100), fontsize=8)
    else:
        ax.scatter(m["ann_vol"] * 100, m["ann_return"] * 100, s=55, color=NAVY)
        ax.annotate(name, (m["ann_vol"] * 100, m["ann_return"] * 100), fontsize=8)
ax.set_xlabel("Annualised volatility (%)")
ax.set_ylabel("Annualised return (%)")
ax.set_title("Risk and return, test window Jan 2022 - Oct 2026")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("charts/risk_return.png", dpi=200)
plt.close(fig)
print("charts saved")
