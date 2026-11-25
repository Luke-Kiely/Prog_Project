"""Panel 1: test every pair of stocks for cointegration and draw the heatmap."""
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.stattools import coint

# The 10 rival pairs from our group plan
RIVAL_PAIRS = [
    ("KO", "PEP"),      # soft drinks
    ("V", "MA"),        # card networks
    ("XOM", "CVX"),     # oil majors
    ("JPM", "BAC"),     # big banks
    ("HD", "LOW"),      # home improvement
    ("WMT", "COST"),    # big-box retail
    ("MCD", "QSR"),     # fast food (QSR=Burger King's Owner)
    ("UPS", "FDX"),     # parcel delivery
    ("NVDA", "AMD"),    # chips
    ("GOOGL", "META"),  # digital advertising
]
TICKERS = [ticker for pair in RIVAL_PAIRS for ticker in pair]

START = "2000-01-01"
END = None                                                    # none up to today                                
PRICE_CACHE = Path(__file__).parent / "data" / "prices.csv"   # data folder next to this file
MIN_SHARED_DAYS = 500                                         # ~2 years of trading days- fewer is too little to test

def load_prices(tickers=TICKERS, start=START, end=END, cache=PRICE_CACHE):
    """Download the latest prices from Yahoo Finance. If that fails, use the saved CSV."""
    try:
        prices = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False)["Close"]
        if prices.empty or prices.isna().all().any():    # nothing back, or a stock missing
            raise ValueError("incomplete download")
        cache.parent.mkdir(exist_ok=True)
        prices.to_csv(cache)                             # keep the backup up to date
    except Exception as error:
        print("Yahoo Finance unavailable, using saved copy:", error)
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    return prices[tickers].dropna(how="all")             # drop days when nothing traded

def hedge_ratio(y, x):
    """How many shares of x to hold against one share of y (slope of y on x)."""
    return y.cov(x) / x.var()

def half_life(spread):
    """Days for a gap in the spread to close halfway. Infinite if it never closes."""
    yesterday = spread.shift(1)                             # spread one day earlier                      
    speed = spread.diff().cov(yesterday) / yesterday.var()  # how fast gaps shrink- (negative=closing)
    if speed >= 0:
        return np.inf
    return -np.log(2) / speed

def test_pair(log_prices, stock_a, stock_b):
    """Return a dictionary of statistics for one pair, or None if too little data."""
    both = log_prices[[stock_a, stock_b]].dropna()     # keep only the days where both stocks traded
    if len(both) < MIN_SHARED_DAYS:                    # need ~2 years of shared prices
        return None
    y, x = both[stock_a], both[stock_b]
    beta = hedge_ratio(y, x)
    t_stat, p_value, critical_values = coint(y, x, maxlag=1, autolag=None)   # Engle-Granger test, 1 lag (fast): small p = tethered
    return {
        "pair": f"{stock_a} / {stock_b}",
        "p_value": p_value,
        "correlation": y.diff().corr(x.diff()),        # do daily returns move together?
        "hedge_ratio": beta,
        "half_life": half_life(y - beta * x),
    }

def screen_all_pairs(log_prices):
    """Test every pair of stocks (190 pairs) and return the results as a DataFrame."""
    tickers = list(log_prices.columns)
    rows = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):           # j > i: each pair only once
            result = test_pair(log_prices, tickers[i], tickers[j])
            if result:                                 # skip pairs with too little data
                rows.append(result)
    return pd.DataFrame(rows)

def make_heatmap(results, cutoff=0.05, view="Both", selected=None):
    """Heatmap of p-values: darker = smaller p, dot = tethered, black box = rival pair."""
    n = len(TICKERS)
    colours = np.full((n, n), np.nan)                           # empty 20x20 grid (diagonal stays blank)
    dots = np.full((n, n), "", dtype=object)                    # a dot on the tethered pairs
    hover = np.full((n, n), "", dtype=object)                   # numbers shown when hovering
    corr = np.full((n, n), np.nan)                              # correlation of daily returns
    for _, row in results.iterrows():
        stock_a, stock_b = row["pair"].split(" / ")
        i, j = TICKERS.index(stock_a), TICKERS.index(stock_b)   # row and column of each stock
        colours[i, j] = colours[j, i] = row["p_value"]          # same pair in both halves
        corr[i, j] = corr[j, i] = row["correlation"]
        hover[i, j] = (f"<b>{row['pair']}</b><br>p-value: {row['p_value']:.4f}"
                       f"<br>Half-life: {row['half_life']:.0f} days")                 
        hover[j, i] = f"<b>{row['pair']}</b><br>Correlation: {row['correlation']:.2f}"   
        if row["p_value"] < cutoff:
            dots[i, j] = dots[j, i] = "●"
    lower = np.tril(np.ones((n, n), dtype=bool))            # True on and below the diagonal
    colours[lower] = np.nan
    dots[lower] = ""  
    corr[np.triu(np.ones((n, n), dtype=bool))] = np.nan     # correlation only in the bottom left  
    fig = go.Figure(go.Heatmap(z=colours, visible=view != "Correlated", x=TICKERS, y=TICKERS,
                               colorscale="Blues_r", zmin=0, zmax=1, xgap=1, ygap=1, colorbar=dict(title="p-value"),
                               customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                               text=dots, texttemplate="%{text}", textfont=dict(color="white")))
    if view != "Tethered":                                      # red/blue layer: correlation (Session 9 style)
        fig.add_trace(go.Heatmap(z=corr, x=TICKERS, y=TICKERS, colorscale="RdBu_r", zmin=-1, zmax=1,
                                 xgap=1, ygap=1, colorbar=dict(title="correlation", x=1.15),
                                 customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    rival_labels = [f"{a} / {b}" for a, b in RIVAL_PAIRS]
    n_tethered = (results[results["pair"].isin(rival_labels)]["p_value"] < cutoff).sum()
    fig.update_layout(title=f"{n_tethered} of the 10 rival pairs are tethered (dot = p < {cutoff:.4g})",
                      height=700, plot_bgcolor="white",
                      xaxis=dict(tickangle=-45, showgrid=False, constrain="domain"),
                      yaxis=dict(autorange="reversed", scaleanchor="x", showgrid=False))
    for stock_a, stock_b in RIVAL_PAIRS:                        # outline the 10 rival pairs
        i, j = TICKERS.index(stock_a), TICKERS.index(stock_b)
        cells = {"Both": [(i, j), (j, i)], "Tethered": [(i, j)], "Correlated": [(j, i)]}[view]
        for r, c in cells:
            fig.add_shape(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                          line=dict(color="black", width=2))
    if selected:                                                # outline the chosen pair in red
        stock_a, stock_b = selected.split(" / ")
        i, j = TICKERS.index(stock_a), TICKERS.index(stock_b)
        for r, c in {"Both": [(i, j), (j, i)], "Tethered": [(i, j)], "Correlated": [(j, i)]}[view]:
            fig.add_shape(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                          line=dict(color="red", width=4))      
    return fig

CACHE = {}                                                      # results already worked out this run

def get_results(first_year=2000, last_year=2026):
    """Screen all pairs for the chosen years, reusing earlier answers so the sliders stay fast."""
    if "prices" not in CACHE:
        CACHE["prices"] = load_prices()                         # download once per run
    if (first_year, last_year) not in CACHE:
        prices = CACHE["prices"].loc[str(first_year):str(last_year)]
        CACHE[(first_year, last_year)] = screen_all_pairs(np.log(prices))
    return CACHE[(first_year, last_year)]

def panel1(first_year=2000, last_year=2026, alpha=0.05, bonferroni=False, view="Both", pair="KO / PEP"):
    """Everything the group dashboard needs: the Panel 1 heatmap for these settings."""
    first_year, last_year = sorted([int(first_year), int(last_year)])
    results = get_results(first_year, last_year)
    if results.empty:
        return go.Figure(layout=dict(title="Not enough data in those years. Pick a longer range."))
    cutoff = alpha / len(results) if bonferroni else alpha     # Bonferroni: divide by the number of tests
    return make_heatmap(results, cutoff, view, pair)

def build_panel1():
    """Panel 1 for the dashboard: the heatmap plus its controls."""
    gr.Markdown("Hover over a square for its numbers. Changing the years re-runs all 190 tests (about 15 seconds).")
    with gr.Row():
        first_year = gr.Slider(2000, 2026, value=2000, step=1, label="From (year)", scale=2)
        last_year = gr.Slider(2000, 2026, value=2026, step=1, label="To (year)", scale=2)
        alpha = gr.Radio([0.01, 0.05, 0.10], value=0.05, label="Significance level", scale=1)
    with gr.Row():
        bonferroni = gr.Checkbox(label="Bonferroni correction (stricter cut-off for 190 tests)")
        view = gr.Radio(["Both", "Tethered", "Correlated"], value="Both", label="Show")
        pair = gr.Dropdown([f"{a} / {b}" for a, b in RIVAL_PAIRS], value="KO / PEP", label="Highlight a pair", visible=False)
    plot = gr.Plot(panel1())                                    # heatmap with default settings
    controls = [first_year, last_year, alpha, bonferroni, view,pair]
    for slider in (first_year, last_year):
        slider.release(panel1, controls, plot)                  # redraw when a slider is let go
    for control in (alpha, bonferroni, view, pair):
        control.change(panel1, controls, plot)                  # instant: results are cached
    return pair    

if __name__ == "__main__":                                                     # only runs when run directly
    prices = load_prices()
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} stocks")
    results = screen_all_pairs(np.log(prices))
    # the 10 most tethered of 190 pairs
    print(results.nsmallest(10, "p_value")[["pair", "p_value", "correlation", "half_life"]].round(4))
    panel1().write_html("heatmap.html")                                        # default settings
    panel1(bonferroni=True, view="Tethered").write_html("heatmap_half.html")   # Bonferroni + half

