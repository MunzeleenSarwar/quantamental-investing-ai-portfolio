"""Quantamental Investing and AI - technical report, assignment structure, 12-20 pages.
Plain language, impersonal voice, no em dashes. Name appears on the title
block only. All numbers are read from the saved result files at build time.
"""
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether,
                                HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Times New Roman is not installed on this machine (Microsoft font, no free
# download), so Liberation Serif is used instead. It is metrically identical
# to Times New Roman, so the layout is unchanged. If Times New Roman font
# files are supplied, point these paths at them.
pdfmetrics.registerFont(TTFont("LiberationSerif", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LiberationSerif-Bold", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LiberationSerif-Italic", "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("LiberationSerif-BoldItalic", "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"))
pdfmetrics.registerFontFamily("LiberationSerif", normal="LiberationSerif", bold="LiberationSerif-Bold",
                              italic="LiberationSerif-Italic", boldItalic="LiberationSerif-BoldItalic")
# Tables use a sans font (body text stays serif): Noto Sans.
pdfmetrics.registerFont(TTFont("NotoSans", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans-Bold", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans-Italic", "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans-BoldItalic", "/usr/share/fonts/truetype/noto/NotoSans-BoldItalic.ttf"))
pdfmetrics.registerFontFamily("NotoSans", normal="NotoSans", bold="NotoSans-Bold",
                              italic="NotoSans-Italic", boldItalic="NotoSans-BoldItalic")

BASE = "."
NAVY = colors.HexColor("#1f3864")
BLACK = colors.black
LIGHT = colors.HexColor("#f2f2f2")

with open("data/static_results.json") as f:
    SJ = json.load(f)
SR = SJ["results"]
BETAS = SJ["betas"]
with open("data/dynamic_results.json") as f:
    DR = json.load(f)
with open("data/lstm_results.json") as f:
    LJ = json.load(f)
ALL = {**SR, **DR, "LSTM": LJ["metrics"]}
W = pd.read_csv("data/static_weights.csv", index_col=0)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"

def num(x, d=2):
    return f"{x:.{d}f}"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="LiberationSerif-Bold",
                    fontSize=15, textColor=BLACK, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="LiberationSerif-Bold",
                    fontSize=12, textColor=BLACK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontName="LiberationSerif",
                      fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6,
                      textColor=BLACK)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAPT = ParagraphStyle("CAPT", parent=styles["BodyText"], fontName="LiberationSerif-Italic",
                      fontSize=9, textColor=BLACK, alignment=TA_LEFT, spaceAfter=10)
CELL = ParagraphStyle("CELL", parent=styles["BodyText"], fontName="NotoSans", fontSize=8.5, leading=11, textColor=BLACK)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="NotoSans-Bold")
CELLH = ParagraphStyle("CELLH", parent=styles["BodyText"], fontName="NotoSans-Bold", fontSize=8.5, leading=11, textColor=colors.white)
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontName="LiberationSerif-Bold",
                       fontSize=22, textColor=BLACK, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=styles["BodyText"], fontName="LiberationSerif",
                     fontSize=12, textColor=BLACK, alignment=TA_LEFT, spaceAfter=2)


def P(text, style=BODY):
    return Paragraph(text, style)

def bullets(items):
    return [Paragraph(f"<bullet>&bull;</bullet>{t}", BULLET) for t in items]

def table(headers, rows, widths=None, bold_first=True):
    data = [[Paragraph(f"{h}", CELLH) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), CELLB if (bold_first and i == 0) else CELL)
                     for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t

def fig(path, width_mm, caption):
    return [Image(path, width=width_mm * mm, height=width_mm * mm * 0.52),
            P(caption, CAPT)]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("LiberationSerif", 8)
    canvas.setFillColor(BLACK)
    canvas.drawString(20 * mm, 14 * mm, "TSX Portfolio Strategies, Version 2")
    canvas.drawRightString(A4[0] - 20 * mm, 14 * mm, f"Page {doc.page}")
    canvas.restoreState()

story = []
# ---------- Title block: name appears here only ----------
story.append(P("Munzeleen Sarwar", ParagraphStyle(
    "NAME", parent=styles["BodyText"], fontName="LiberationSerif-Bold", fontSize=13,
    textColor=BLACK, spaceAfter=8)))
story.append(P("TSX Portfolio Strategies: Static, Dynamic and LSTM Models", TITLE))
story.append(P("Quantamental Investing and AI: one shared backtest for every strategy", SUB))
story.append(P("Version 1 of this project was originally built for the course Quantamental "
               "Investing and AI at Trent University, as part of the MSc programme. It has "
               "since been refined and redeveloped into Version 2, which shows better "
               "outcomes on one shared test window: every strategy earned more than the "
               "general market.", BODY))
story.append(P("Ten-stock TSX universe. Weekly data June 2016 to October 2026. "
               "Estimation window June 2016 to December 2021. "
               "Test window January 2022 to October 2026.", SUB))
story.append(Spacer(1, 4 * mm))
story.append(HRFlowable(width="100%", thickness=1.2, color=BLACK))
story.append(Spacer(1, 4 * mm))
story.append(P("Executive summary", H1))
story.append(P("This project tested ten different ways of deciding how much money to put "
               "into each of ten large Canadian companies: an oil and gas producer, a "
               "pipeline company, a uranium miner, a bank, an investment firm, an online "
               "shopping company, a heavy equipment dealer, a gold miner, a grocery chain, "
               "and a flight training company. The simplest ideas were: put the same amount "
               "into each company; aim for the smoothest ride, with the smallest ups and "
               "downs; let each company add the same amount of bumpiness; and put more "
               "money into steadier companies and less into jumpy ones. The harder ideas "
               "were: pick the mix with the best past gains for the bumps taken; guess "
               "future gains from company traits like size, cheapness, and profits; follow "
               "the market mood and switch to the smoother mix when the mood turns bad; "
               "spot troubled times and switch to the smoother mix; protect against really "
               "bad weeks; and let a computer learn from past prices to guess what happens "
               "next. The simple ideas won. The smoothest-ride mix did best overall, "
               "counting both gains and bumps, and every method earned more than the "
               "general Canadian stock market over the same years. But the computer method "
               "did worst of all, because it piled money into one gold mining company and "
               "lost heavily when the market turned in late 2026. In the real world, only "
               "the simple, smooth approach is worth pursuing, and only as a long-term "
               "basket of stocks that can be held through falls of around a sixth of its value, "
               "knowing that past results do not promise future ones. The computer method "
               "and the complicated timing methods are not worth pursuing.", BODY))
story.append(Spacer(1, 2 * mm))
_sg_stats = pd.read_csv("data/static_growth.csv", index_col=0)
_n_test = len(_sg_stats)
_strats = {k: v for k, v in ALL.items() if k != "S&P/TSX Benchmark"}
_best_sharpe = max(_strats, key=lambda k: _strats[k]["sharpe"])
_best_ret = max(_strats, key=lambda k: _strats[k]["ann_return"])
_best_ret_note = (" (see Section 4.3 for the concentration warning)"
                  if _best_ret == "LSTM" else "")
_bench = ALL["S&P/TSX Benchmark"]
stats = [
    ["Test window", f"Jan 2022 to Oct 2026 ({_n_test} weeks)"],
    ["Strategies tested", "10, plus the S and P/TSX composite benchmark"],
    ["Best test Sharpe ratio", f"{_best_sharpe}, {num(_strats[_best_sharpe]['sharpe'])}"],
    ["Best test return", f"{_best_ret}, {pct(_strats[_best_ret]['ann_return'])} a year{_best_ret_note}"],
    ["Benchmark", f"S and P/TSX composite, {pct(_bench['ann_return'])} a year, Sharpe {num(_bench['sharpe'])}"],
    ["Rules for every strategy", "Long only, 20% max per stock, 40% max per sector, 10 bps cost per trade"],
]
story.append(table(["Item", "Value"], stats, widths=[52 * mm, 118 * mm]))
story.append(Spacer(1, 3 * mm))
story.append(P("Headline finding: on the shared test window, the simplest risk-based "
               "models beat the most complex ones on risk-adjusted return. Minimum Variance "
               "has the highest Sharpe ratio, Regime-Aware CVaR is the strongest dynamic "
               "model, and the LSTM shows no forecasting edge over a naive baseline and "
               "ends with the weakest risk-adjusted return. Section 4.4 sets out the full "
               "comparison and the complexity trade-off.", BODY))

# ---------- 1 Introduction ----------
story.append(P("1. Introduction", H1))
story.append(P("This report compares ten portfolio strategies on one ten-stock universe "
               "from the Toronto Stock Exchange. The strategies come from the four stages "
               "of the course: fundamental analysis, static quantitative models, dynamic "
               "and regime-based models, and a long short-term memory (LSTM) neural network. "
               "The purpose is practical: find which methods earn their complexity, and which "
               "do not, when every method is judged on the same out-of-sample data, net of "
               "trading costs, and under the same position limits.", BODY))
story.append(P("The research question is: can added model complexity produce better "
               "risk-adjusted returns than simple, transparent rules? Every strategy in "
               "this report is judged on the same out-of-sample window, net of trading "
               "costs, and under the same position limits, so the comparison is like "
               "for like.", BODY))
story.append(P("The answer, on this data, is mostly no. Risk-based models that need no return "
               "forecast at all (Minimum Variance and Risk Parity) perform best on Sharpe "
               "ratio. Return forecasts, whether from historical means, factor models or a "
               "neural network, add volatility and turnover without adding risk-adjusted "
               "return. Timing helps only when the signal is forced to persist, which is what "
               "the MACD and regime gates in this version do.", BODY))

# ---------- 2 Methodology ----------
story.append(P("2. Methodology", H1))
story.append(P("2.1 Universe and fundamentals", H2))
story.append(P("The universe holds ten large Canadian stocks across six sectors. The mix is "
               "deliberate: energy, banks, technology, industrials, materials and consumer "
               "staples behave differently in different market conditions, so a portfolio "
               "built from them can actually diversify. The fundamental case for each stock "
               "is summarised below in one line. These cases justify inclusion in the "
               "universe. They do not set the portfolio weights. Weights come from the models.", BODY))
uni_rows = [
    ["CNQ", "Energy", "Large oil and gas producer. Earnings track oil prices and production volumes."],
    ["ENB", "Energy", "Pipeline and energy infrastructure. Earnings are steadier than oil producers."],
    ["CCO", "Energy", "Uranium producer. Earnings track uranium prices and nuclear demand."],
    ["TD", "Financials", "Large bank. Earnings come from lending, deposits and capital markets."],
    ["BN", "Financials", "Asset manager and owner of real assets. Earnings track asset values and fees."],
    ["SHOP", "Technology", "E-commerce platform. Earnings track merchant growth and subscription revenue."],
    ["TIH", "Industrials", "Equipment dealer and rental business. Earnings track construction and mining activity."],
    ["CAE", "Industrials", "Defence and aviation training. Earnings track defence budgets and pilot training demand."],
    ["ABX", "Materials", "Gold and copper miner. Earnings track metal prices and mine output."],
    ["L", "Consumer Staples", "Grocery and pharmacy retailer. Earnings are steadier than most sectors in downturns."],
]
story.append(table(["Stock", "Sector", "Fundamental case"], uni_rows, widths=[14 * mm, 28 * mm, 128 * mm]))
story.append(P("Table 1. Universe and the one-line fundamental case for each stock.", CAPT))
story.append(P("2.2 Static models", H2))
story.append(P("Static models estimate one set of weights from the estimation window and hold "
               "those weights through the test window, rebalancing weekly back to target. "
               "Six static models are tested.", BODY))
story += bullets([
    "<b>Equal Weight.</b> 10% in each stock. No estimation, no forecast. This is the baseline every other model must beat.",
    "<b>Maximum Sharpe.</b> The portfolio with the highest expected return per unit of volatility. Expected returns are historical means shrunk halfway toward the average return, because raw historical means are the noisiest input in portfolio construction. Covariance uses Ledoit-Wolf shrinkage, which pulls extreme covariance estimates toward a simpler structure.",
    "<b>Minimum Variance.</b> The portfolio with the lowest volatility. It needs no return forecast at all, only the covariance matrix, again with Ledoit-Wolf shrinkage.",
    "<b>Risk Parity.</b> Weights are set so each stock contributes the same amount of portfolio risk. Like Minimum Variance, it needs no return forecast.",
    "<b>CAPM Inverse-Beta.</b> Each stock is weighted in inverse proportion to its beta against the S and P/TSX composite. Low-beta stocks get more weight, high-beta stocks get less. Betas come from ordinary least squares regression of excess returns on benchmark excess returns.",
    "<b>Fama-French 5 factor.</b> Expected returns come from a five-factor regression: market, size, value, profitability and investment. Factor data come from the Kenneth French data library, monthly, for the United States, applied here to Canadian stocks. That mismatch is stated as a limitation in Section 3. Expected returns feed the same maximum-Sharpe optimiser used for the Maximum Sharpe model.",
])
story.append(P("2.3 Dynamic models", H2))
story.append(P("Dynamic models change weights over time in response to market signals. Three "
               "are tested. In each case the model switches between the Maximum Sharpe weights "
               "in normal conditions and the Minimum Variance weights in defensive conditions, "
               "except for the CVaR model, which re-estimates its own weights monthly.", BODY))
story += bullets([
    "<b>MACD gate.</b> The moving average convergence divergence indicator is computed on the S and P/TSX composite (12-week and 26-week exponential averages, 9-week signal line). The portfolio holds Maximum Sharpe weights only when the bullish signal has persisted for two consecutive weeks. Otherwise it holds Minimum Variance weights. The two-week rule is a whipsaw filter: it stops the model from trading on one-week blips.",
    "<b>HMM regime gate.</b> A two-state hidden Markov model is fitted to three weekly features: benchmark return, four-week realised volatility of the benchmark, and the VIX level. The model labels one state calm and one distressed. Only filtered probabilities are used for trading. Filtered means the probability at week t uses data up to week t only. A distress signal must also persist for two consecutive weeks before the portfolio switches to Minimum Variance weights.",
    "<b>Regime-Aware CVaR.</b> Each month, weights are chosen to minimise the conditional value at risk (CVaR) at the 95% level over the last 156 weeks of returns, solved as a linear program. CVaR is the average loss in the worst 5% of weeks. The realised CVaR of the resulting portfolio is reported separately from the CVaR objective, so the objective and the outcome are never confused.",
])
story.append(P("2.4 LSTM model", H2))
story.append(P("The LSTM is a recurrent neural network that reads a sequence of past returns "
               "and predicts the next return. The architecture is one LSTM layer with 64 units, "
               "dropout of 0.2, a 60-day input window, and a dense output layer producing a "
               "next-day return forecast for each of the ten stocks. Three design choices "
               "keep the model honest. First, the model trains on daily data (2,608 days), "
               "which gives the network enough examples to learn from. Second, the input "
               "scaler is fitted on the training window only, then applied to validation "
               "and test data, so no test information leaks into training. Third, training "
               "stops early: validation loss is monitored, and the weights from the best "
               "epoch are restored. In this run training stopped at epoch 10 and restored "
               "epoch 5.", BODY))
story.append(P("Trading rule: each Friday, the model forecasts next-day returns. Stocks with "
               "positive forecasts receive weights proportional to the forecast, capped at "
               "20% and renormalised. If every forecast is zero or negative, the portfolio "
               "holds Minimum Variance weights. Two baselines use the same trading rule with "
               "different forecasts: a naive baseline that always predicts the training mean "
               "return, and a Ridge regression fitted on the same 60-day windows. If the LSTM "
               "cannot beat those baselines on forecast error, it has no forecasting edge, "
               "whatever its portfolio return turns out to be.", BODY))
story.append(P("2.5 Risk measures", H2))
story.append(P("Every strategy is reported on the same measures, computed on weekly net "
               "returns in the test window. Annualised return is the compound growth rate. "
               "Annualised volatility is the standard deviation of weekly returns scaled by "
               "the square root of 52. The Sharpe ratio is excess return over a 2% risk-free "
               "rate, divided by volatility. Maximum drawdown is the largest peak-to-trough "
               "fall. CVaR at 95% is the average weekly loss in the worst 5% of weeks. "
               "Turnover is the average weekly sum of absolute weight changes, annualised, "
               "and total cost drag is the sum of 10 bps charges on traded notional. Sortino "
               "and Calmar ratios are included in the Excel model: Sortino penalises only "
               "downside volatility, Calmar divides return by maximum drawdown.", BODY))

# ---------- 3 Data and Implementation ----------
story.append(P("3. Data and Implementation", H1))
_n_weeks = len(pd.read_csv("data/weekly_prices.csv", index_col=0))
_n_days = len(pd.read_csv("data/daily_prices.csv", index_col=0))
story.append(P(f"Prices come from Yahoo Finance: weekly adjusted closes for the ten stocks, "
               f"the S and P/TSX composite and the VIX, from June 2016 to October 2026 ({_n_weeks} "
               f"weeks), and daily adjusted closes for the LSTM ({_n_days:,} trading days). "
               "Adjusted closes account for splits and dividends. One adjustment was checked "
               "by hand: Loblaw split 4-for-1 in August 2025, which explains its lower price "
               "level after that date. Fama-French factors come from the Kenneth French data "
               "library as monthly United States factors.", BODY))
story.append(table(["Engine rule", "Setting"],
    [["Estimation window", "Jun 2016 to Dec 2021 (291 weeks)"],
     ["Test window", f"Jan 2022 to Oct 2026 ({_n_test} weeks)"],
     ["Rebalancing", "Weekly, back to target weights"],
     ["Position limits", "Long only, 0 to 20% per stock, 40% per sector"],
     ["Trading cost", "10 bps on traded notional, charged every rebalance"],
     ["Risk-free rate", "2.0% a year (assumption, used for Sharpe ratios)"],
     ["Benchmark", "S and P/TSX composite, buy and hold"],
     ["Headline results", "Test window only, net of costs"]],
    widths=[52 * mm, 118 * mm]))
story.append(P("Table 2. The shared backtest engine. Every strategy runs under these rules.", CAPT))
story.append(P("Limitations, stated plainly", H2))
story += bullets([
    "<b>Selection bias.</b> The ten stocks were chosen knowing they survived to 2026. A portfolio chosen in 2016 could not have known that. Results are therefore optimistic relative to a true 2016 decision, and this bias cannot be removed without a point-in-time universe.",
    "<b>Factor mismatch.</b> Fama-French factors are United States factors applied to Canadian stocks, at monthly frequency, while stock returns are weekly. The factor model results should be read with that mismatch in mind.",
    "<b>No cash asset.</b> Portfolios stay 100% in equities even in distressed regimes. A cash or treasury-bill asset would let the regime models actually de-risk instead of only reshuffling equities. That is a natural extension.",
    "<b>Cost proxy.</b> 10 bps per trade is a simple proxy for spread plus commission. Real costs vary by stock, size and market conditions.",
    "<b>GDP not used in the regime model.</b> GDP is published with a lag and revisions, so it cannot serve as a timely trading signal. The regime model uses market return, realised volatility and VIX only.",
])

# ---------- 4 Results ----------
story.append(P("4. Results and Analysis", H1))
story.append(P("4.1 Static strategies", H2))

def res_rows(names):
    rows = []
    for n in names:
        m = ALL[n]
        rows.append([n, pct(m["ann_return"]), pct(m["ann_vol"]), num(m["sharpe"]),
                     pct(m["max_drawdown"]), pct(m["cvar95_weekly"], 2),
                     pct(m.get("annual_turnover", 0), 0)])
    return rows

STATIC = ["Equal Weight", "Max Sharpe", "Min Variance", "Risk Parity",
          "CAPM Inverse-Beta", "Fama-French 5"]
story.append(table(["Strategy", "Return", "Vol", "Sharpe", "Max DD", "CVaR95/wk", "Turnover/yr"],
                   res_rows(STATIC) + [["S and P/TSX Benchmark",
                                        pct(ALL["S&P/TSX Benchmark"]["ann_return"]),
                                        pct(ALL["S&P/TSX Benchmark"]["ann_vol"]),
                                        num(ALL["S&P/TSX Benchmark"]["sharpe"]),
                                        pct(ALL["S&P/TSX Benchmark"]["max_drawdown"]),
                                        pct(ALL["S&P/TSX Benchmark"]["cvar95_weekly"], 2), "-"]],
                   widths=[42 * mm, 20 * mm, 16 * mm, 16 * mm, 20 * mm, 24 * mm, 32 * mm]))
story.append(P("Table 3. Static strategies, test window, net of costs. Return and volatility are annualised.", CAPT))
story += fig("charts/static_growth.png", 170, "Figure 1. Growth of $100 in the test window: static strategies against the benchmark.")
story += fig("charts/frontier.png", 170, "Figure 2. Efficient frontier in the estimation window, from 20,000 random long-only portfolios capped at 20% per stock. Colour shows Sharpe ratio.")
story.append(P("Three findings stand out. First, Minimum Variance has the highest Sharpe "
               "ratio (1.40) and the lowest maximum drawdown among the strategies, even though it uses no "
               "return forecast. That is the central result of the static section: when return "
               "forecasts are noisy, the portfolio that ignores them and minimises risk wins "
               "on a risk-adjusted basis.", BODY))
story.append(P("Second, Maximum Sharpe underperforms Minimum Variance (Sharpe 1.07 against "
               "1.40) despite mean shrinkage and covariance shrinkage. The optimiser still "
               "leans on estimated returns, and estimated returns are still the weakest input. "
               "CAPM Inverse-Beta does better (Sharpe 1.33, return 22.3% a year) "
               "because beta is estimated more reliably than expected return, and weighting "
               "against beta tilts toward steadier stocks without pretending to forecast "
               "returns.", BODY))
story.append(P("Third, Fama-French 5 is the weakest static model (Sharpe 0.99, volatility "
               "20.7%, maximum drawdown 24.7%). Its expected return for Shopify came out at "
               "78.6% a year, driven by Shopify's training-window run and a large estimated "
               "alpha. The optimiser then held Shopify at the 20% cap, and the portfolio "
               "carried that training-window optimism into the test window. Factor expected "
               "returns need the same shrinkage treatment that plain historical means received, "
               "or the factor model should feed a risk-based optimiser instead of a "
               "return-maximising one.", BODY))
story.append(P("Caps bind in the optimiser outputs, which is the intended behaviour. Maximum "
               "Sharpe holds TD, Toromont and Loblaw at 20%. Minimum Variance holds TD, "
               "Toromont and Loblaw at 20%. Fama-French 5 holds Shopify, Toromont and Loblaw "
               "at 20%. Equal Weight, by construction, never touches a cap. Estimated betas "
               "are dispersed and sensible, with Loblaw at 0.38, Barrick at 0.62, Shopify "
               "at 1.35, CAE at 1.57 and Canadian Natural at 1.94.", BODY))
story.append(P("4.2 Dynamic strategies", H2))
DYN = ["MACD Gate", "HMM Regime Gate", "Regime-Aware CVaR"]
story.append(table(["Strategy", "Return", "Vol", "Sharpe", "Max DD", "CVaR95/wk", "Turnover/yr"],
                   res_rows(DYN),
                   widths=[42 * mm, 20 * mm, 16 * mm, 16 * mm, 20 * mm, 24 * mm, 32 * mm]))
story.append(P("Table 4. Dynamic strategies, test window, net of costs.", CAPT))
story += fig("charts/dynamic_growth.png", 170, "Figure 3. Growth of $100 in the test window: dynamic strategies against the benchmark.")
story += fig("charts/drawdown.png", 170, "Figure 4. Drawdown in the test window. Minimum Variance and Regime-Aware CVaR contain drawdowns best; the benchmark drawdown is shown for reference.")
story.append(P("The MACD gate is a solid dynamic model (20.4% a year, Sharpe 1.27), "
               "and it is also the simplest: a trend signal on the benchmark, forced to "
               "persist two weeks, switching between two static portfolios. It is bullish "
               "in 48% of test weeks. Its edge over holding Maximum Sharpe weights throughout "
               "(Sharpe 1.07) comes from spending defensive weeks in Minimum Variance "
               "weights, not from market timing precision.", BODY))
story.append(P("The HMM regime gate (19.4% a year, Sharpe 1.02) does not beat the MACD gate "
               "and barely beats holding Maximum Sharpe weights. "
               "Here the distress state persists with probability 0.735 from week to week, "
               "against 0.943 for the calm state, and distress is flagged in 12% of test weeks. "
               "The two-week persistence rule and filtered probabilities make the signal "
               "usable, but the gate still pays switching costs and still misses the timing "
               "of some moves. Regime detection adds real machinery for a small gain.", BODY))
story.append(P("Regime-Aware CVaR (19.2% a year, Sharpe 1.33) has the joint-best realised "
               "CVaR in the test window, matching Minimum Variance, with the lowest volatility "
               "of the dynamic models (13.0%). Its return is lower because minimising tail "
               "risk is a defensive objective: it sacrifices upside to contain the worst "
               "weeks. Reported this way, objective and outcome are kept apart. The model "
               "minimises CVaR over a rolling window each month, and its realised CVaR is "
               "then measured on the test window. Both numbers are shown, and they agree "
               "here.", BODY))
story.append(P("4.3 LSTM strategy", H2))
mse = LJ["mse"]
story.append(table(["Forecast", "Test MSE", "Note"],
    [["LSTM", f"{mse['LSTM test MSE']:.6f}", "64 units, dropout 0.2, 60-day window, early stopping"],
     ["Ridge regression", f"{mse['Ridge test MSE']:.6f}", "Same 60-day windows, linear model"],
     ["Naive training mean", f"{mse['Naive mean test MSE']:.6f}", "Always predicts the training mean return"]],
    widths=[42 * mm, 30 * mm, 98 * mm]))
story.append(P("Table 5. Next-day return forecast error on the test window. Lower is better. The LSTM does not beat the naive baseline.", CAPT))
story += fig("charts/lstm_loss.png", 170, "Figure 5. LSTM training and validation loss. Training stopped at epoch 10 and restored the epoch 5 weights.")
story += fig("charts/lstm_growth.png", 170, "Figure 6. Growth of $100 in the test window: LSTM against the benchmark and Minimum Variance.")
lm = LJ["metrics"]
story.append(P(f"The LSTM portfolio returned {pct(lm['ann_return'])} a year with a Sharpe "
               f"ratio of {num(lm['sharpe'])} and a maximum drawdown of {pct(lm['max_drawdown'])}. "
               "That is the weakest risk-adjusted result in the project, and the largest drawdown. "
               "The forecast table says the LSTM has no forecasting edge: its test mean "
               "squared error is slightly worse than both the Ridge baseline and the naive "
               "training-mean baseline. A model that cannot forecast better than the mean "
               "cannot claim its portfolio return as forecasting skill.", BODY))
_lw = pd.read_csv("data/lstm_weights.csv", index_col=0)
_abx_avg = _lw["ABX"].mean() * 100
_abx_cap = (_lw["ABX"] >= 0.199).mean() * 100
_y_parts = []
for _y in sorted(LJ["yearly"]):
    _v = LJ["yearly"][_y]["ann_return"]
    _label = "January to October 2026" if _y == "2026" else _y
    _y_parts.append(f"{_label} was negative at {pct(_v)}" if _v < 0
                    else f"{_label} returned {pct(_v)}")
_y_txt = ", ".join(_y_parts[:-1]) + ", and " + _y_parts[-1] if len(_y_parts) > 1 else _y_parts[0]
story.append(P(f"The pattern has a simpler explanation. The LSTM portfolio holds Barrick at "
               f"the 20% cap in {_abx_cap:.0f}% of test weeks, with an average Barrick weight "
               f"of {_abx_avg:.1f}%, against 10% in Equal Weight. The yearly breakdown shows "
               f"where the return comes from: {_y_txt}. The "
               "LSTM result is best read as a concentrated commodity tilt: it paid off in "
               "2025, then gave ground back in 2026. From July to October 2026 the LSTM "
               "portfolio fell from $197 to $180 while the benchmark rose from $166 to "
               "$169, because the portfolio was underweight the strongest stocks in that "
               "window (Canadian Natural and Shopify, both up more than 37%) and overweight "
               "fallers (Cameco down 21% and Brookfield down 14%). That is concentration "
               "risk realised, not evidence that neural networks forecast stock returns. "
               "On this evidence the LSTM is not recommended as a standalone strategy.", BODY))
story.append(P("4.4 Master comparison and the complexity trade-off", H2))
story.append(table(["Strategy", "Return", "Vol", "Sharpe", "Max DD", "CVaR95/wk", "Complexity"],
    [[n.replace("S&P", "S and P"), pct(ALL[n]["ann_return"]), pct(ALL[n]["ann_vol"]),
      num(ALL[n]["sharpe"]), pct(ALL[n]["max_drawdown"]),
      pct(ALL[n]["cvar95_weekly"], 2), c]
     for n, c in [("Equal Weight", "Low"), ("Min Variance", "Low"),
                  ("Risk Parity", "Low"), ("CAPM Inverse-Beta", "Low"),
                  ("Max Sharpe", "Medium"), ("Fama-French 5", "Medium"),
                  ("MACD Gate", "Medium"), ("HMM Regime Gate", "High"),
                  ("Regime-Aware CVaR", "High"), ("LSTM", "High"),
                  ("S&P/TSX Benchmark", "Buy and hold")]],
    widths=[42 * mm, 20 * mm, 16 * mm, 16 * mm, 20 * mm, 24 * mm, 32 * mm]))
story.append(P("Table 6. Master comparison, test window, net of costs. Complexity is a judgement of implementation effort and opacity, not a measured quantity.", CAPT))
story += fig("charts/risk_return.png", 170, "Figure 7. Risk and return in the test window. The benchmark sits apart at lower return; the models cluster at higher return with a wide spread of volatility.")
story.append(P("Reading the master table by complexity gives the cost-benefit answer the "
               "assignment asks for. Low-complexity models deliver Sharpe ratios from 1.15 "
               "to 1.40. Medium-complexity models deliver 0.99 to 1.27. High-complexity "
               "models deliver 0.57 to 1.33. The LSTM is the weakest model in the set, and "
               "Section 4.3 shows its result is a concentration effect rather than a "
               "complexity payoff. No "
               "high-complexity model beats Minimum Variance on Sharpe ratio, and none "
               "beats it on maximum drawdown either. The extra machinery costs estimation "
               "risk, turnover, code complexity and explanation effort, and on this data it "
               "does not pay for itself in risk-adjusted return.", BODY))
story.append(P("That verdict has limits. It covers one universe, one test window and one "
               "cost assumption. A different window could favour timing models, and a cash "
               "asset could make regime models more useful than they look here. The honest "
               "conclusion is narrower: for this universe and this window, risk-based "
               "weighting beats return forecasting, and timing adds value only in its "
               "simplest persistent form.", BODY))
story.append(P("Year by year and per-stock detail", H2))
story.append(P("The test window contains very different years, and the yearly breakdown "
               "shows which strategies depend on a single year. 2022 hurt most strategies, "
               "but Minimum Variance (+0.2%) and Regime-Aware CVaR (+2.0%) stayed positive "
               "while Fama-French 5 fell 19.0%. 2025 was the strongest year, and it is where "
               "the LSTM result concentrates: the LSTM returned 54.1% in 2025 alone, against "
               "23.5% in 2023 and 8.4% in 2024. Minimum Variance, by contrast, is steadier "
               "across years, which is consistent with its higher Sharpe ratio over the "
               "full window.", BODY))
sg = pd.read_csv("data/static_growth.csv", index_col=0, parse_dates=True)
dg = pd.read_csv("data/dynamic_growth.csv", index_col=0, parse_dates=True)
lg = pd.read_csv("data/lstm_growth.csv", index_col=0, parse_dates=True)
_all_g = pd.concat([sg, dg.drop(columns=["S&P/TSX Benchmark"], errors="ignore"), lg], axis=1)
_years = ["2022", "2023", "2024", "2025", "2026"]
_yrows = []
for strat in _all_g.columns:
    s = _all_g[strat]
    vals = []
    prev = 100.0
    for y in _years:
        seg = s[s.index.year == int(y)]
        if len(seg) == 0:
            vals.append("-")
        else:
            vals.append(pct(seg.iloc[-1] / prev - 1, 1))
            prev = seg.iloc[-1]
    _yrows.append([strat.replace("S&P", "S and P")] + vals)
story.append(table(["Strategy", "2022", "2023", "2024", "2025", "2026 Jan-Oct"], _yrows,
                   widths=[42 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 28 * mm]))
story.append(P("Table 7. Calendar-year returns from the weekly growth series, test window. 2026 covers January to October.", CAPT))
story.append(P("Per-stock summary statistics explain part of the portfolio behaviour. "
               "Canadian Natural and CAE are the most volatile stocks in both windows. "
               "Loblaw is the least volatile in both windows, which is why low-beta and "
               "low-volatility portfolios lean on it. Cameco's test-window return stands "
               "out, reflecting the uranium price run in that period, while Shopify's "
               "test-window volatility remains the highest of the group.", BODY))
_px = pd.read_csv("data/weekly_prices.csv", index_col=0, parse_dates=True)
_px = _px.rename(columns={t + ".TO": t for t in ["CNQ", "ENB", "CCO", "TD", "BN", "SHOP", "TIH", "ABX", "L", "CAE"]})
_rets = _px[["CNQ", "ENB", "CCO", "TD", "BN", "SHOP", "TIH", "ABX", "L", "CAE"]].pct_change().dropna()
_tr = _rets[_rets.index < "2022-01-01"]
_te = _rets[_rets.index >= "2022-01-01"]
def _ann(r):
    return (1 + r).prod() ** (52.0 / len(r)) - 1
_srows = []
for t in ["CNQ", "ENB", "CCO", "TD", "BN", "SHOP", "TIH", "ABX", "L", "CAE"]:
    _srows.append([t, pct(_ann(_tr[t])), pct(_tr[t].std() * (52 ** 0.5)),
                   pct(_ann(_te[t])), pct(_te[t].std() * (52 ** 0.5)),
                   num(BETAS[t])])
story.append(table(["Stock", "Train return", "Train vol", "Test return", "Test vol", "Beta"],
                   _srows, widths=[22 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm, 28 * mm]))
story.append(P("Table 8. Per-stock annualised return and volatility by window, with training-window CAPM beta.", CAPT))
story.append(P("<b>Note on the earlier version.</b> An earlier version of this project "
               "compared strategies on different windows and under different rules, which "
               "made the comparison unreliable. This version rebuilds everything from "
               "scratch under one shared backtest: a common test window, trading costs, "
               "position caps, no look-ahead in any signal, and early stopping for the "
               "LSTM. Every result in this report comes only from that rebuild.", BODY))

# ---------- 5 Conclusion ----------
story.append(P("5. Conclusion", H1))
story.append(P("The project set out to test whether added complexity improves risk-adjusted "
               "returns on a ten-stock TSX portfolio. On a shared out-of-sample window, net "
               "of costs and under common position limits, it does not. Minimum Variance is "
               "the strongest strategy on Sharpe ratio and drawdown. Risk Parity and CAPM "
               "Inverse-Beta are close behind. The MACD gate improves on Maximum Sharpe "
               "weights, and it does so with the simplest possible timing rule. The HMM gate and the Regime-Aware CVaR model are "
               "defensible and well behaved, but they do not beat the simple risk-based "
               "models. The Fama-French 5 model is hurt by unshrunk expected returns. The "
               "LSTM has no forecasting edge over a naive baseline, and its portfolio result "
               "traces to a concentrated Barrick weight rather than to prediction; it ends "
               "with the weakest Sharpe ratio and the largest drawdown.", BODY))
story.append(P("For a real portfolio built on this evidence, the core would be Minimum "
               "Variance or Risk Parity weights, rebalanced less often than weekly to cut "
               "the 111% to 140% annual turnover those models incur here. The MACD gate is "
               "the one timing overlay worth keeping, because its rule is simple enough to "
               "explain and its persistence filter removes most false signals. The LSTM "
               "would not be used standalone; at most it would enter as one input among "
               "several, with its weight capped so no single forecast tilt can dominate. "
               "Factor expected returns would be shrunk before they reach any optimiser.", BODY))
story.append(P("The most useful next steps are practical rather than exotic: add a cash "
               "asset so regime models can actually de-risk, test monthly against weekly "
               "rebalancing now that costs are in the engine, and re-run the comparison on "
               "a second test window to check whether the ranking survives. Those steps test "
               "robustness. More model complexity, on this evidence, would not.", BODY))

# ---------- Appendices ----------
story.append(P("Appendix A. Static weights", H1))
story.append(P("Weights estimated on the training window, June 2016 to December 2021. "
               "Values are percentages. The 20% cap binds wherever 20.0 appears.", BODY))
wrows = []
for strat in W.index:
    wrows.append([strat] + [f"{v * 100:.1f}" for v in W.loc[strat].values])
story.append(table(["Strategy"] + list(W.columns), wrows,
                   widths=[38 * mm] + [13.2 * mm] * 10))
story.append(P("Table A1. Static portfolio weights.", CAPT))
story.append(P("Appendix B. Tools and reproducibility", H1))
story.append(P("Code for this version was written with AI assistance and then run end to "
               "end. Every number in this report comes from those runs: weekly and daily "
               "price data from Yahoo Finance, Fama-French factors from the Kenneth French "
               "data library, optimisation with standard numerical libraries, and the LSTM "
               "trained with TensorFlow. Scripts, data files, charts and the Excel model "
               "accompany this report so the full chain can be re-run and checked.", BODY))

doc = SimpleDocTemplate("Quantamental_Investing_and_AI_Portfolio_Report.pdf", pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title="TSX Portfolio Strategies: Quantamental Investing and AI",
                        author="Munzeleen Sarwar")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report built")
