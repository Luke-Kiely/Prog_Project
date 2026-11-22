"""Panel 1 calculations: download prices and test every pair of rivals."""
from pathlib import Path

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
END = None                                # none up to today                                
PRICE_CACHE = Path("data") / "prices.csv"
MIN_SHARED_DAYS = 500                     # ~2 years of trading days- fewer is too little to test

def load_prices(tickers=TICKERS, start=START, end=END, cache=PRICE_CACHE):
    """Download the latest prices from Yahoo Finance. If that fails, use the saved CSV."""
    try:
        prices = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False)["Close"]
        if prices.empty or prices.isna().all().any():   # nothing back, or a stock missing
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
    t_stat, p_value, critical_values = coint(y, x)     # cointegration(Engle-Granger) test:small p=tethered
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

def make_heatmap(results, cutoff=0.05, half=False):
    """Grid of p-values for every pair: darker = smaller p-value."""
    n = len(TICKERS)
    colours = np.full((n, n), np.nan)                           # empty 20x20 grid (diagonal stays blank)
    dots = np.full((n, n), "", dtype=object)                    # a dot on the tethered pairs
    for _, row in results.iterrows():
        stock_a, stock_b = row["pair"].split(" / ")
        i, j = TICKERS.index(stock_a), TICKERS.index(stock_b)   # row and column of each stock
        colours[i, j] = colours[j, i] = row["p_value"]          # same pair in both halves
        if row["p_value"] < cutoff:
            dots[i, j] = dots[j, i] = "●"
    if half:                                                    # hide repeated half
        lower = np.tril(np.ones((n, n), dtype=bool))            # True on and below the diagonal
        colours[lower] = np.nan
        dots[lower] = ""    
    fig = go.Figure(go.Heatmap(z=colours, x=TICKERS, y=TICKERS,
                               colorscale="Blues_r", zmin=0, zmax=1, xgap=1, ygap=1,
                               text=dots, texttemplate="%{text}", textfont=dict(color="white")))
    fig.update_layout(title="Which pairs are tethered?", height=700,
                      yaxis=dict(autorange="reversed", scaleanchor="x"))
    return fig

if __name__ == "__main__":                                              # only runs when run directly
    prices = load_prices()
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} stocks")
    results = screen_all_pairs(np.log(prices))
    # the 10 most tethered of 190 pairs
    print(results.nsmallest(10, "p_value")[["pair", "p_value", "correlation", "half_life"]].round(4))
    make_heatmap(results).write_html("heatmap.html")                    # open this file to check the heatmap
    make_heatmap(results, half=True).write_html("heatmap_half.html")    # upper half only
