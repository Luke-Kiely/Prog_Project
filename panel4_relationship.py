"""Panel 4 - Relationship over time (Charlie)."""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PRICE_FILE = Path("data") / "prices.csv"  # run from the project folder
START = "2000-01-01"  # download start date (one year before FIRST_YEAR for the hedge ratio)

# Each pair ordered [Stock A, Stock B] with Stock A on y-axis and Stock B on x-axis
PAIRS = {
    "Coca-Cola vs PepsiCo": ["KO", "PEP"],
    "Visa vs Mastercard": ["V", "MA"],
    "ExxonMobil vs Chevron": ["XOM", "CVX"],
    "JPMorgan vs Bank of America": ["JPM", "BAC"],
    "Home Depot vs Lowe's": ["HD", "LOW"],
    "Walmart vs Costco": ["WMT", "COST"],
    "McDonald's vs Burger King": ["MCD", "QSR"],
    "UPS vs FedEx": ["UPS", "FDX"],
    "Nvidia vs AMD": ["NVDA", "AMD"],
    "Alphabet vs Meta": ["GOOGL", "META"],
}
TICKERS = [ticker for pair in PAIRS.values() for ticker in pair]  # all tickers in one list

# Settings
HEDGE_WINDOW = 252  # trading days used for the hedge ratio (about 1 year)
SIGNAL_Z = 2.5  # a signal day has a z-score that is beyond plus/minus 2.5
Z_CAP = 3  # colour scale runs from -3 to +3
MIN_DAYS = 20  # a year needs at least this many trading days to be used
FIRST_YEAR = 2001  # starts at 2001 as hedge ratio needs a full year of data to start
PLAY_SECONDS = 1.0  # seconds per year while playing


def load_prices() -> pd.DataFrame:
    """Gets the daily closing prices for all tickers, either from a saved CSV or by downloading."""
    if PRICE_FILE.exists():
        print("Using saved copy:", PRICE_FILE)
        return pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    try:
        prices = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    except Exception as err:
        raise RuntimeError(f"Could not download prices ({err}).") from err
    prices = prices.dropna(axis=1, how="all")  # drops tickers that failed to download
    if len(prices.columns) == len(TICKERS):  # only saves if every ticker is downloaded
        PRICE_FILE.parent.mkdir(exist_ok=True)
        prices.to_csv(PRICE_FILE)
        print("Downloaded once and saved:", PRICE_FILE)
    return prices


# Load the data
PRICES = load_prices()  # loaded once when the panel starts
print("Prices loaded (days, tickers):", PRICES.shape)
LAST_YEAR = PRICES.index.max().year  # last year in the data, not today's year