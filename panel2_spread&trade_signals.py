"""Panel 2 - Spread & Trade Signals. Top: A's price vs the fair value implied by B. It marks the trades. Bottom: z-score of the gap"""

import gradio as gr
import numpy as np
import pandas as pd
import yfinance as yf 
from plotly.subplots import make_subplots

RIVAL_PAIRS = [("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("JPM", "BAC"), ("HD", "LOW"), ("WMT", "COST"), ("MCD", "QSR"), ("UPS", "FDX"), ("NVDA", "AMD"), ("GOOGL", "META")] 

WINDOW = 60
ENTRY_Z = 2.0
STOP_Z = 3.5
COLOURS = {"a": "#2a78d6", "fair": "#898781", "long": "#0ca30c", "short": "#d03b3b", "ink": "#52514e"}

def load_prices():
  tickers = [t for pair in RIVAL_PAIRS for t in pair]
  return yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
  
def build_signals(prices, a, b):
  df = np.log(prices[[a,b]].dropna())
  roll_a, roll_b = df[a].rolling(WINDOW), df[b].rolling(WINDOW)
  cov = roll_a.cov(df[b])
  beta = cov / roll_b.var()
  fit = pd.DataFrame({"beta": beta, "alpha": roll_a.mean() - beta * roll_b.mean(), "sd": np.sqrt(roll_a.var() - beta * cov)}).shift(1)
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
  return (f"<b>{action}</b><br>Entry: {t.entry:%d %b %Y}<br>" f"Exit: {t.exit:%d %b %Y}{' (stop-loss)' if t.stopped else''}<br>" f"Held: {t.days} days<br><b>P&L: {t.pnl:+.1%}</b>")
                                                                        
def make_figure(label, year_from, year_to):
  a, b = label.split(" / ")
  sig, trades = pair_data(a, b)
  year_from, year_to = sorted([int(year_from), int(year_to)])
  sig = sig.loc[str(year_from):str(year_to)]
  trades = trades[trades.entry.between(sig.index.min(), sig.index.max())]
  if trades.empty:
    title = f"<b>{a} vs {b}</b> | no trades in year {year_from}-{year_to}"
  else: 
    title = (f"<b>{a} vs {b}</b> | {year_from}-{year_to} | {len(trades)} trades |" f"{(trades.pnl > 0).mean():.0%} winners | " f"{(1 + trades.pnl).prod() - 1:+.1%} total return (before costs)")
  text = [hover_text(t, a, b) for t in trades.itertuples()]
  is_long = trades.side == 1

  fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38])
  fig.add_scatter(x=sig.index, y=sig.fair, name=f"{a} fair value (based on {b})", customdata=sig[b], line=dict(color=COLOURS["fair"], width=1.5, dash="dot"), row=1, col=1, hovertemplate=f"Fair value: $%{{y:.2f}}<br>{b}: $%{{customdata:.2f}}<extra></extra>")
  fig.add_scatter(x=sig.index, y=sig[a], name=f"{a} price", line=dict(color=COLOURS["a"], width=2), fill="tonexty", fillcolor="rgba(42,120,214,0.12)", row=1, col=1, hovertemplate=f"{a}: $%{{y:.2f}}<extra></extra>")
  fig.add_scatter(x=trades.entry, y=sig[a].reindex(trades.entry), mode="markers", name="Entry (▲ buy / ▼ short)", text=text, hovertemplate="%{text}<extra></extra>", marker=dict(size=12, line=dict(width=1.5, color="white"), symbol=np.where(is_long, "triangle-up", "triangle-down"), color=np.where(is_long, COLOURS["long"], COLOURS["short"])), row=1, col=1)
  fig.add_scatter(x=trades.exit, y=sig[a].reindex(trades.exit), mode="markers", name="Exit", text=text, hovertemplate="%{text}<extra></extra>", marker=dict(size=9, color="white", line=dict(width=2, color=COLOURS["ink"])), row=1, col=1)
  fig.add_scatter(x=sig.index, y=sig.z, name="Z-score", showlegend=False, line=dict(color=COLOURS["ink"], width=1.5), hovertemplate="z = %{y:.2f}<extra></extra>", row=2, col=1)
  for y, colour, label, pos in [(ENTRY_Z, COLOURS["short"], f"{a} expensive: short {a}", "top left"), (-ENTRY_Z, COLOURS["long"], f"{a} cheap: buy {a}", "bottom left"), (0, COLOURS["fair"], "", "top left")]:
    fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dash" if y else "solid"), row=2, col=1, annotation_text=label, annotation_position=pos, annotation_font_color=colour)
  fig.update_layout(title=dict(text=title, x=0.01), height=620, template="plotly_white", hovermode="closest", margin=dict(t=90, l=60, r=20, b=40), legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", traceorder="normal"))
  fig.update_yaxes(title_text=f"{a} price", tickprefix="$", row=1, col=1)
  fig.update_yaxes(title_text="Gap (z-score)", zeroline=False, row=2, col=1)
  return fig

LABELS = [f"{a} / {b}" for a, b in RIVAL_PAIRS]

def build_panel2():
  first, last = load_prices().index[0].year, load_prices().index[-1].year
  pair = gr.Dropdown(LABELS, value=LABELS[0], label="Rival Pair")
  plot = gr.Plot(make_figure(LABELS[0], last - 2, last))
  with gr.Row():
    year_from = gr.Slider(first, last, value=last - 2, step=1, label="From (year)")
    year_to = gr.Slider(first, last, value=last, step=1, label="To (year)")
  pair.change(make_figure, [pair, year_from, year_to], plot)
  for slider in (year_from, year_to):
    slider.release(make_figure, [pair, year_from, year_to], plot)

if __name__ == "__main__":
  print("Downloading prices...")
  print(f"Got {load_prices().shape[1]} tickers, {len(load_prices())} days. Building first chart... ")
  with gr.Blocks(title="Panel 2 - Spread & Trade Signals") as demo: 
    build_panel2()
  demo.launch()
  

  

