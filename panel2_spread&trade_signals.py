"""Panel 2 - Spread & Trade Signals. Top: A's price vs the fair value implied by B. It marks the trades. Bottom: z-score of the gap"""

import numpy as np
import pandas as pd
import yfinance as yf 
from dash import Dash, Input, Output, Patch, State, callback, dcc, html, no_update
from plotly.subplots import make_subplots

RIVAL_PAIRS = [("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("JPM", "BAC"), ("HD", "LOW"), ("WMT", "COST"), ("MCD", "QSR"), ("UPS", "FDX"), ("NVDA", "AMD"), ("GOOGL", "META")] 

WINDOW = 60
ENTRY_Z = 2.0
STOP_Z = 3.5
COLOURS = {"a": "#2a78d6", "fair": "#898781", "long": "#0ca30c", "short": "#d03b3b", "ink": "#522514e"}

def load_prices():
  tickers = [t for pair in RIVAL_PAIRS for t in pair]
  return yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
  
def build_signals(prices, a, b):
  df = np.log(prices[[a,b]].dropna())
  roll_a, roll_b = df[a].rolling(WINDOW), df[b].rolling(WINDOW)
  cov = roll_a.cov(df[b])
  beta = cov / roll_b.var()
  fit = pd.DataFrame({"beta": fit.beta, "z": (df[a] - fair_log) / fit.sd, "fair": np.exp(fair_log)})
  fair_log = fit.alpha + fit.beta * df[b]
  sig = pd.DataFrame({"beta": fit.beta, "z": (df[a] - fair_log) / fit.sd, "fair": np.exp(fair_log)})
  return sig.join(prices[[a,b,]]).dropna()

def find_trades(sig, a, b): 
  trades, side, start, blocked = [], 0, None, False
  for day, row in sig.iterrows():
    if abs(row.z) < ENTRY_Z:
      blocked = False
    if side == 0 and not blocked and ENTRY_Z < abs(row.z) < STOP_Z: 
      side = -1 if row.z > 0 else 1
      start = day 
    elif side != 0 and (side * row.z >= 0 or abs(row.z) > STOP_Z or day == sig.index[-1]):
      entry = sig.loc[start]
      pnl = side * ((row[a] / entry[a] - 1) - entry.beta * (row[b] / entry [b] - 1))
      trades.append({"side": side, "entry": start, "exit": day, "days": (day - start).days, "pnl": pnl / (1 + abs(entry.beta)), "stopped": abs(row.z) > STOP_Z})
      side, blocked = 0, abs(row.z) > STOP_Z
  return pd.DataFrame(trades, columns=["side", "entry", "exit", "days", "pnl", "stopped"])
  
def pair_data(a, b): 
  sig = build_signals(load_prices(), a, b)
  return sig, find_trades(sig, a, b)

def hover_text(t, a, b):
  action = f"Buy {a}, Short {b}" if t.side == 1 else f"Short {a}, Buy {b}"
  return (f"<b>(actions)</b><br>Entry: {t/entry:%d %b %Y}<br>" f"Exit: {t.exit: %d %b %Y}{' (stop-loss)' if t.stopped else''}<br>" f"Held: {t.days} days<br><b>P&L: {t.pnl:+.1%}</b>")
                                                                        
def view_stats(a, b, start, end): 
  sig, trades = pair_data(a, b)
  visible = sig.loc[start:end, [a, "fair"]]
  y_range = [float(visible.min().min() * 0.97), float(visible.max().max() * 1.03)] if len(visible) else no_update
  shown = trades[(trades.entry >= visible.index.min()) & (trades.entry <= visible.index.max())]
  if shown.empty: 
    return f"<b>{a} vs {b}</b> | no trades in this period", y_range
  total = (1 + shown.pnl).prod() - 1
  return (f"<b>{a} vs {b}</b> | {len(shown)} trades | {(shown.pnl > 0).mean():.0%} winners | "
          f"{total: +.1%} total return (before costs)"), y_range

