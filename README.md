# Quantamental Investing and AI: TSX Portfolio Strategies

Quantitative portfolio project on a 10-stock TSX universe, with one shared
backtest for every strategy.

## Universe

CNQ (Canadian Natural Resources), ENB (Enbridge), CCO (Cameco), TD (TD Bank),
BN (Brookfield Corp), SHOP (Shopify), TIH (Toromont), ABX (Barrick Mining),
L (Loblaw), CAE (CAE). Benchmark: S and P/TSX composite.

## Backtest rules (same for every strategy)

- Estimation: Jun 2016 to Dec 2021. Test: Jan 2022 to Oct 2026 (251 weeks).
- Long only, 20% max per stock, 40% max per sector, weights sum to 100%.
- Weekly rebalancing. Trading cost: 10 bps on traded notional.
- Risk-free rate: 2.0% a year (assumption, used for Sharpe ratios).
- Headline results are test-window only, net of costs.

## Strategies

Static: Equal Weight, Max Sharpe (shrunk means, Ledoit-Wolf covariance),
Min Variance, Risk Parity, CAPM Inverse-Beta, Fama-French 5.
Dynamic: MACD gate, HMM regime gate (filtered probabilities, 2-week persistence),
Regime-Aware CVaR (monthly min-CVaR).
Machine learning: LSTM (64 units, dropout 0.2, 60-day window, daily data,
train-only scaler, early stopping) plus Ridge and naive baselines.

## Headline results, test window, net of costs

| Strategy | Return | Sharpe | Max drawdown |
|---|---|---|---|
| Min Variance | 20.2% | 1.40 | -16.7% |
| CAPM Inverse-Beta | 22.3% | 1.33 | -19.5% |
| Risk Parity | 21.4% | 1.29 | -19.1% |
| MACD Gate | 20.4% | 1.27 | -20.2% |
| LSTM | 12.9% | 0.57 | -31.7% |
| S and P/TSX Benchmark | 11.4% | 0.73 | -16.7% |

Honest LSTM note: the LSTM does not beat a naive baseline on forecast error
(test MSE 0.000492 vs naive 0.000488). Its portfolio result traces to a
Barrick weight at the 20% cap in 86% of test weeks, not to forecasting skill,
and it ends with the weakest Sharpe ratio and the largest drawdown.

## Files

- `Quantamental_Investing_and_AI_Portfolio_Report.pdf` — 12-page technical report (assignment structure).
- `Quantamental_Investing_and_AI_Portfolio_Model.xlsx` — universe, weights, master results, growth series, betas.
- `analysis.ipynb` — executed notebook reproducing the results.
- `scripts/` — `engine.py`, `static_models.py`, `dynamic_models.py`,
  `lstm_model.py`, `charts.py`, `build_model.py`, `build_report.py`,
  `build_notebook.py`. Run in that order (engine first, report last).
- `data/` — weekly and daily prices, weights, results JSON files.
- `charts/` — report charts.

## Limitations

Universe chosen with knowledge of survival to 2026 (selection bias).
Fama-French factors are United States monthly factors applied to Canadian
stocks. No cash asset: portfolios stay 100% in equities. Costs are a 10 bps
proxy. GDP was not used in the regime model (lagged publication, so it cannot serve
as a timely trading signal).
