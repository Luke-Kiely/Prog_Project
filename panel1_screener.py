"""Panel 1 calculations: download prices and test every pair of rivals."""
from pathlib import Path

import numpy as np
import pandas as pd
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
END = None                                
PRICE_CACHE = Path("data") / "prices.csv"
MIN_SHARED_DAYS = 500                     # a pair needs approx 2 years of prices in common

def load_prices(tickers=TICKERS, start=START, end=END, cache=PRICE_CACHE, refresh=False):
    """Return daily closing prices: download once, then reuse the saved CSV."""
    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        prices = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False)["Close"]
        cache.parent.mkdir(exist_ok=True)
        prices.to_csv(cache)
    return prices[tickers].dropna(how="all")   # drop days when nothing traded

def hedge_ratio(y, x):
    """How many shares of x to hold against one share of y (slope of y on x)."""
    return y.cov(x) / x.var()

def half_life(spread):
    """Days for a gap in the spread to close halfway. Infinite if it never closes."""
    yesterday = spread.shift(1)                       
    speed = spread.diff().cov(yesterday) / yesterday.var()
    if speed >= 0:
        return np.inf
    return -np.log(2) / speed

def test_pair(log_prices, stock_a, stock_b):
    """Return a dictionary of statistics for one pair, or None if too little data."""
    both = log_prices[[stock_a, stock_b]].dropna()     # days when both traded
    if len(both) < MIN_SHARED_DAYS:                                # need ~2 years of shared prices
        return None
    y, x = both[stock_a], both[stock_b]
    beta = hedge_ratio(y, x)
    t_stat, p_value, critical_values = coint(y, x)     # cointegration test
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
            if result:
               rows.append(result)
    return pd.DataFrame(rows)

if __name__ == "__main__":                            # only runs when run directly
    prices = load_prices()
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} stocks")
    results = screen_all_pairs(np.log(prices))
    print(results.nsmallest(10, "p_value")[["pair", "p_value", "correlation", "half_life"]].round(4))
