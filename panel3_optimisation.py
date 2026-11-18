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