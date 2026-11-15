from pathlib import Path

import pandas as pd
import yfinance as yf

#The 10 rival pairs from our group plan
RIVAL_PAIRS=[
    ("KO", "PEP"),      # soft drinks
    ("V", "MA"),        # card networks
    ("XOM", "CVX"),     # oil majors
    ("JPM", "BAC"),     # big banks
    ("HD", "LOW"),      # home improvement
    ("WMT", "COST"),    # big-box retail
    ("MCD", "QSR"),     # fast food (QSR = Burger King's owner)
    ("UPS", "FDX"),     # parcel delivery
    ("NVDA", "AMD"),    # chips
    ("GOOGL", "META"),  # digital advertising
]

TICKERS= [ticker for pair in RIVAL_PAIRS for ticker in pair]

START= "2000-01-01"
END= None
PRICE_CACHE= Path("data") / "prices.csv"

def load_prices(tickers=TICKERS, start=START, end=END, cache=PRICE_CACHE, refresh=False):
    """Return daily closing prices: download once, then reuse thesaved CSV. """
    if cache.exists() and not refresh:
        prices=pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        prices=yf.download(tickers, start=start, end=end,
                           auto_adjust=True, progress=False)["Close"]
        cache.parent.mkdir(exist_ok=True)
        prices.to_csv(cache)
    return prices[tickers]


def clean_prices(prices):
    """Drop days when no stock traded, but keep the gaps before a stock listed."""
    return prices.dropna(how="all")

def print_data_check(raw_prices, clean):
    """Print what was downloaded and when each stock starts."""
    print("=== DATA CHECK ===")
    print(f"Downloaded: {raw_prices.shape[0]} days x {raw_prices.shape[1]} stocks")
    print(f"From {clean.index.min().date()} to {clean.index.max().date()}")
    print("First date with a price, per stock:")
    print(clean.apply(lambda col: col.first_valid_index().date()).to_string())

raw_prices = load_prices()
prices = clean_prices(raw_prices)
print_data_check(raw_prices, prices)
