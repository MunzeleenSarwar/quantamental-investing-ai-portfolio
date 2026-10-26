"""Build the executed analysis notebook for Quantamental Investing and AI."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

cells = []
cells.append(nbf.v4.new_markdown_cell(
    "# Quantamental Investing and AI: TSX Portfolio Strategies\n\n"
    "Ten-stock TSX universe (CNQ, ENB, CCO, TD, BN, SHOP, TIH, ABX, L, CAE). "
    "One shared backtest: estimation Jun 2016-Dec 2021, test Jan 2022-Oct 2026, "
    "long only, 20% cap per stock, 40% cap per sector, 10 bps cost per trade.\n\n"
    "This notebook reproduces the static and dynamic results. The LSTM is trained "
    "in `scripts/lstm_model.py` (TensorFlow) and its saved results are loaded here."))
cells.append(nbf.v4.new_code_cell(
    "import sys, json\nsys.path.insert(0, 'scripts')\n"
    "import pandas as pd, numpy as np\n"
    "from engine import SHORT, load_weekly, cap_weights, backtest, metrics\n"
    "px, rets, bench, vix = load_weekly('.')\n"
    "train = rets[rets.index < '2022-01-01']\n"
    "test = rets[rets.index >= '2022-01-01']\n"
    "print('train weeks:', len(train), '| test weeks:', len(test))\n"
    "print('stocks:', SHORT)"))
cells.append(nbf.v4.new_markdown_cell("## Static weights (estimated on the train window)"))
cells.append(nbf.v4.new_code_cell(
    "w = pd.read_csv('data/static_weights.csv', index_col=0)\n"
    "display((w * 100).round(1))"))
cells.append(nbf.v4.new_markdown_cell("## Master results, test window, net of costs"))
cells.append(nbf.v4.new_code_cell(
    "import json\n"
    "with open('data/static_results.json') as f: sr = json.load(f)['results']\n"
    "with open('data/dynamic_results.json') as f: dr = json.load(f)\n"
    "with open('data/lstm_results.json') as f: lr = json.load(f)\n"
    "allres = {**sr, **dr, 'LSTM': lr['metrics']}\n"
    "rows = []\n"
    "for n, m in allres.items():\n"
    "    rows.append({'Strategy': n, 'Ann return': round(m['ann_return']*100, 1),\n"
    "                 'Ann vol': round(m['ann_vol']*100, 1), 'Sharpe': round(m['sharpe'], 2),\n"
    "                 'Max DD': round(m['max_drawdown']*100, 1),\n"
    "                 'CVaR95/wk': round(m['cvar95_weekly']*100, 2)})\n"
    "display(pd.DataFrame(rows).sort_values('Sharpe', ascending=False))"))
cells.append(nbf.v4.new_markdown_cell("## LSTM forecast check: does it beat the naive baseline?"))
cells.append(nbf.v4.new_code_cell(
    "print('MSE table:', lr['mse'])\n"
    "print('LSTM epochs (early stopping):', lr['epochs'], '| val loss:', round(lr['val_loss'], 4))\n"
    "print('LSTM yearly test returns:', {k: round(v['ann_return']*100, 1) for k, v in lr['yearly'].items()})"))
cells.append(nbf.v4.new_markdown_cell("## Growth charts"))
cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "sg = pd.read_csv('data/static_growth.csv', index_col=0, parse_dates=True)\n"
    "dg = pd.read_csv('data/dynamic_growth.csv', index_col=0, parse_dates=True)\n"
    "lg = pd.read_csv('data/lstm_growth.csv', index_col=0, parse_dates=True)\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "for c in ['Min Variance', 'CAPM Inverse-Beta', 'MACD Gate']:\n"
    "    s = sg[c] if c in sg else dg[c]\n"
    "    ax.plot(s.index, s, label=c)\n"
    "ax.plot(lg.index, lg['LSTM'], label='LSTM')\n"
    "ax.plot(sg.index, sg['S&P/TSX Benchmark'], label='Benchmark', ls='--')\n"
    "ax.set_title('Growth of $100, test window'); ax.legend(); ax.grid(alpha=0.3)\n"
    "plt.show()"))
nb.cells = cells
nbf.write(nb, "analysis.ipynb")
print("notebook written")
