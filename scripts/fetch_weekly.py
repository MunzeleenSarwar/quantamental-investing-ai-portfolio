"""Fetch weekly prices through Oct 2026, one ticker at a time with retries."""
import time
import pandas as pd
import yfinance as yf

TICKERS = ["CNQ.TO", "ENB.TO", "CCO.TO", "TD.TO", "BN.TO",
           "SHOP.TO", "TIH.TO", "ABX.TO", "L.TO", "CAE.TO",
           "^GSPTSE", "^VIX"]
frames = {}
for t in TICKERS:
    ok = False
    for attempt in range(4):
        try:
            h = yf.Ticker(t).history(start="2016-06-01", end="2026-11-01",
                                     interval="1wk", auto_adjust=True)
            if len(h) > 400:
                idx = h.index
                try:
                    idx = idx.tz_localize(None)
                except TypeError:
                    pass
                h.index = pd.DatetimeIndex(idx).normalize()
                frames[t] = h["Close"]
                print(f"{t}: {len(h)} rows to {h.index[-1].date()}", flush=True)
                ok = True
                break
            print(f"{t}: attempt {attempt} returned {len(h)} rows, retrying", flush=True)
        except Exception as e:
            print(f"{t}: attempt {attempt} error {type(e).__name__}, retrying", flush=True)
        time.sleep(5)
    if not ok:
        print(f"{t}: FAILED after retries", flush=True)
px = pd.DataFrame(frames).sort_index()
stocks = [t for t in TICKERS if not t.startswith("^")]
px = px.dropna(subset=stocks)
px[["^GSPTSE", "^VIX"]] = px[["^GSPTSE", "^VIX"]].ffill()
print("combined rows:", len(px), "from", px.index[0].date(), "to", px.index[-1].date())
print("remaining NaNs:", int(px.isna().sum().sum()))
px.to_csv("data/weekly_prices.csv")
print("saved data/weekly_prices.csv")
