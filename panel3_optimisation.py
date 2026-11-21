from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Panel 3 - Strategy Optimisation
# Does the best pairs trading setting from past data still work in future data?

PRICE_FILE = Path("data") / "prices.csv"

PAIRS = {
    "KO vs PEP": ["KO", "PEP"],
    "V vs MA": ["V", "MA"],
    "XOM vs CVX": ["XOM", "CVX"],
    "JPM vs BAC": ["JPM", "BAC"],
    "HD vs LOW": ["HD", "LOW"],
    "WMT vs COST": ["WMT", "COST"],
    "MCD vs QSR": ["MCD", "QSR"],
    "UPS vs FDX": ["UPS", "FDX"],
    "NVDA vs AMD": ["NVDA", "AMD"],
    "GOOGL vs META": ["GOOGL", "META"],
}
TICKERS = [ticker for pair in PAIRS.values() for ticker in pair]

ENTRIES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]    # z-score needed to open a trade
WINDOWS = [10, 20, 40, 60, 120, 250]        # lookback in trading days
TRADING_DAYS = 252
MIN_TRADES = 10                             # a best setting needs at least this many trades
SHARPE_LIMIT = 1.5                          # fixed scale so pairs can be compared


# Step 1: get the prices

def load_prices():
    """Download prices from Yahoo Finance, or use the saved copy if that fails."""
    try:
        prices = yf.download(TICKERS, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        missing = [ticker for ticker in TICKERS if prices[ticker].isna().all()]
        if missing:
            raise ValueError(f"no prices for {missing}")
        print("Using live prices from Yahoo Finance")
    except Exception as e:
        print("Yahoo Finance unavailable:", e)
        prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
        print("Using the saved copy in", PRICE_FILE)
    return prices


# Step 2: the trading strategy

def get_spread(prices, split):
    """Return the log price spread and the hedge ratio (fitted on in-sample days only)."""
    stock_a, stock_b = prices.columns
    log_a = np.log(prices[stock_a])
    log_b = np.log(prices[stock_b])
    hedge_ratio = log_a.iloc[:split].cov(log_b.iloc[:split]) / log_b.iloc[:split].var()
    return log_a - hedge_ratio * log_b, hedge_ratio


def get_positions(zscore, entry):
    """Return the position each day: 1 = long the spread, -1 = short, 0 = no trade."""
    position = 0
    positions = []
    for z in zscore:
        if position == 0 and z > entry:
            position = -1           # spread too wide, bet it closes
        elif position == 0 and z < -entry:
            position = 1            # spread too narrow, bet it widens
        elif position == 1 and z >= 0:
            position = 0            # back to normal so close the trade
        elif position == -1 and z <= 0:
            position = 0
        positions.append(position)
    return pd.Series(positions, index=zscore.index)


def run_strategy(spread, window, entry, cost_bps):
    """Return the position held, the daily move in the spread and the trading costs."""
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    held = get_positions(zscore, entry).shift(1)        # trade the day after the signal
    move = spread - spread.shift(1)
    costs = (held - held.shift(1)).abs() * cost_bps / 10000
    return held, move, costs


def get_sharpe(returns):
    """Return the annualised Sharpe ratio (risk-free rate taken as 0)."""
    if returns.std() == 0:
        return 0                    # no trades, avoid dividing by zero
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)