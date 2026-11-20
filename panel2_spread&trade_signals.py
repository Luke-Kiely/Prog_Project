"""Panel 2 - Spread & Trade Signals. Top: A's price vs the fair value implied by B. It marks the trades. Bottom: z-score of the gap"""

import numpy as np
import pandas as pd
import yfinance as yf 
from dash import Dash, Input, Output, Patch, State, callback, dcc, html, no_update
from plotly.subplots import make_subplots

""" need to put in importing pairs of data piece"""

""" need to put in the load prices function""" 

def builid_signals(prices, a, b):
  df = np.log(prices[[a,b]].dropna())
  roll_a, roll_b = df[a].rolling(WINDOW), df[b].rolling(WINDOW)
  cov = roll_a.cov(df[b])
  beta = cov / roll_b.var()
  fit = pd.DataFrame({"beta": fit.beta, "z": (df[a] - fair_log) / fit.sd, "fair": np.exp(fair_log)})
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
  return pd.DateFrame(trades, columns=["side", "entry", "exit", "days", "pnl", "stopped"])
  
