"""Panel 2 - Spread & Trade Signals. Top: A's price vs the fair value implied by B. It marks the trades. Middle: z-score of the gap. Bottom: P&L on each trade. The three charts share one time axis. The panel also has a plain-English guide to the charts and controls."""

from functools import lru_cache
from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
import yfinance as yf 
from plotly.subplots import make_subplots

RIVAL_PAIRS = [("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("JPM", "BAC"), ("HD", "LOW"), ("WMT", "COST"), ("MCD", "QSR"), ("UPS", "FDX"), ("NVDA", "AMD"), ("GOOGL", "META")] 

WINDOW = 60
ENTRY_Z = 2.0
STOP_Z = 3.5
PERIODS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "All": None}
COLOURS = {"a": "#2a78d6", "fair": "#898781", "long": "#0ca30c", "short": "#d03b3b", "ink": "#d9d9d6"}

@lru_cache(maxsize=1)
def load_prices():
  tickers = [t for pair in RIVAL_PAIRS for t in pair]
  try: 
    prices = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    if prices.empty or prices.isna().all().any():
      raise ValueError("incomplete download")
    return prices
  except Exception as error:
    print("Yahoo Finance unavailable, using saved copy:", error)
    return pd.read_csv(Path(__file__).parent / "data" / "prices.csv", index_col=0, parse_dates=True)
  
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
    elif side != 0 and (side * row.z >= 0 or abs(row.z) > STOP_Z):
      entry = sig.loc[start]
      pnl = side * ((row[a] / entry[a] - 1) - entry.beta * (row[b] / entry [b] - 1))
      trades.append({"side": side, "entry": start, "exit": day, "days": (day - start).days, "pnl": pnl / (1 + abs(entry.beta)), "stopped": abs(row.z) > STOP_Z})
      side, blocked = 0, abs(row.z) > STOP_Z
  return pd.DataFrame(trades, columns=["side", "entry", "exit", "days", "pnl", "stopped"])

@lru_cache(maxsize=None)
def pair_data(a, b): 
  sig = build_signals(load_prices(), a, b)
  return sig, find_trades(sig, a, b)

def window(label, period):
  a, b = label.split(" / ")
  sig, trades = pair_data(a, b)
  if PERIODS[period]:
    sig = sig.loc[sig.index[-1] - pd.DateOffset(years=PERIODS[period]):]
  trades = trades[trades.entry.between(sig.index.min(), sig.index.max())]
  return a, b, sig, trades

def hover_text(t, a, b):
  action = f"Buy {a}, Short {b}" if t.side == 1 else f"Short {a}, Buy {b}"
  return (f"<b>{action}</b><br>Entry: {t.entry:%d %b %Y}<br>" f"Exit: {t.exit:%d %b %Y}{' (stop-loss)' if t.stopped else''}<br>" f"Held: {t.days} days<br><b>P&L: {t.pnl:+.1%}</b>")
                                                                        
def make_figure(label, period):
  a, b, sig, trades = window(label, period)
  dates = f"{sig.index.min():%b %Y} - {sig.index.max():%b %Y}" if len(sig) else "no data"
  if trades.empty:
    title = f"<b>{a} vs {b}</b> | {dates}<br> No Trades in this period"
  else: 
    title = (f"<b>{a} vs {b}</b> | {dates}<br> | {len(trades)} trades | " f"{(trades.pnl > 0).mean():.0%} winners | " f"{(1 + trades.pnl).prod() - 1:+.1%} total return (before costs)")
  text = [hover_text(t, a, b) for t in trades.itertuples()]
  is_long = trades.side == 1
  running_total = (1 + trades.pnl).cumprod() - 1

  fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.5, 0.27, 0.23], subplot_titles=(f"1. {a} Price vs Its Fair Value Based On {b}", "2. How Stretched The Gap is (Z-Score)", "3. Profit or Loss On Each Trade"))
  fig.add_scatter(x=sig.index, y=sig.fair, name=f"{a} fair value (based on {b})", customdata=sig[b], line=dict(color=COLOURS["fair"], width=1.5, dash="dot"), row=1, col=1, hovertemplate=f"Fair value: $%{{y:.2f}}<br>{b}: $%{{customdata:.2f}}<extra></extra>")
  fig.add_scatter(x=sig.index, y=sig[a], name=f"{a} price", line=dict(color=COLOURS["a"], width=2), fill="tonexty", fillcolor="rgba(42,120,214,0.12)", row=1, col=1, hovertemplate=f"{a}: $%{{y:.2f}}<extra></extra>")
  fig.add_scatter(x=trades.entry, y=sig[a].reindex(trades.entry), mode="markers", name="Entry (▲ buy / ▼ short)", text=text, hovertemplate="%{text}<extra></extra>", marker=dict(size=12, line=dict(width=1.5, color="white"), symbol=np.where(is_long, "triangle-up", "triangle-down"), color=np.where(is_long, COLOURS["long"], COLOURS["short"])), row=1, col=1)
  fig.add_scatter(x=trades.exit, y=sig[a].reindex(trades.exit), mode="markers", name="Exit", text=text, hovertemplate="%{text}<extra></extra>", marker=dict(size=9, color="white", line=dict(width=2, color=COLOURS["ink"])), row=1, col=1)
  fig.add_scatter(x=sig.index, y=sig.z, name="Z-score", showlegend=False, line=dict(color=COLOURS["ink"], width=1.5), hovertemplate="z = %{y:.2f}<extra></extra>", row=2, col=1)
  for y, colour, label, pos in [(ENTRY_Z, COLOURS["short"], f"{a} expensive: short {a}", "top left"), (-ENTRY_Z, COLOURS["long"], f"{a} cheap: buy {a}", "bottom left"), (0, COLOURS["fair"], "", "top left")]:
    fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dash" if y else "solid"), row=2, col=1, annotation_text=label, annotation_position=pos, annotation_font_color=colour, annotation_bgcolor="rgba(17, 17, 17,0.85)")
  fig.add_bar(x=trades.exit, y=trades.pnl, name="Trade P&L", showlegend=False, customdata=running_total, marker_color=np.where(trades.pnl > 0, COLOURS["long"], COLOURS["short"]), hovertemplate="Trade P&L: %{y:+.1%}<br>Running Total: %{customdata:+.1%}<extra></extra>", row=3, col=1)
  fig.add_hline(y=0, line=dict(color=COLOURS["fair"], width=1), row=3, col=1)
  fig.update_layout(title=dict(text=title, x=0.01, y=0.97, yanchor="top"), height=760, template="plotly_dark", hovermode="closest", margin=dict(t=110, l=60, r=20, b=40), legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", traceorder="normal"))
  fig.update_annotations(selector=dict(xref="paper"), x=0, xanchor="left", font=dict(size=13, color=COLOURS["ink"]))
  fig.update_xaxes(showline=True, mirror=True, linecolor="#d9d9d6")
  fig.update_yaxes(showline=True, mirror=True, linecolor="#d9d9d6")
  fig.update_yaxes(title_text=f"{a} price", tickprefix="$", row=1, col=1)
  fig.update_yaxes(title_text="Gap (z-score)", zeroline=False, row=2, col=1)
  fig.update_yaxes(title_text="P&L Per Trade", tickformat="+.0%", zeroline=False, row=3, col=1)
  return fig

LABELS = [f"{a} / {b}" for a, b in RIVAL_PAIRS]

def pick_period(label, chosen): 
  buttons = [gr.Button(variant="primary" if p == chosen else "secondary") for p in PERIODS]
  return [chosen, make_figure(label, chosen)] + buttons

def explain(label):
  """Plain-English guide to the three charts, written for the selected pair."""
  a, b = label.split(" / ")
  return f"""### How to read this panel
**The idea:** {a} and {b} are rivals, so their prices usually move together. When {a} drifts too far from where {b} says it should be, we bet on the gap closing.

#### 1. Price vs fair value
- **Blue line:** {a}'s actual price.
- **Grey dotted line:** {a}'s *fair value*, what {a} "should" cost given {b}'s price. It comes from a rolling {WINDOW}-day regression of log({a}) on log({b}), giving a hedge ratio (β) and an intercept (α). It only uses data up to the previous day, so there is no look-ahead bias.
- **Shaded area:** the *spread*, how far {a} has drifted from fair value.
- **▲ green:** {a} is cheap, so buy {a} and short β units of {b}.
- **▼ red:** {a} is expensive, so short {a} and buy β units of {b}.
- **○ white:** the trade is closed.
- **Hover** over any marker for the direction, dates, days held and P&L.

#### 2. Z-score: how stretched is the gap?
- **z = spread ÷ its typical size** over the last {WINDOW} days, so it measures how *unusual* today's gap is.
- **Above +{ENTRY_Z:g}** (red dashed): {a} is unusually expensive, which is a short signal.
- **Below −{ENTRY_Z:g}** (green dashed): {a} is unusually cheap, which is a buy signal.
- **Back to 0:** the gap has closed, so take profit.
- **Beyond ±{STOP_Z:g}:** stop-loss. The gap kept widening, so cut the trade, with no new entry until z falls back inside ±{ENTRY_Z:g}.

#### 3. Profit or loss per trade
- One bar per completed trade, placed at its exit date: **green** is a win, **red** is a loss.
- Return on both legs combined (gross exposure), **before costs**. Hover over a bar for the running total.

#### Headline
- The number of trades, % winners and compounded total return for the **selected period**. Trades still open are not counted.

#### Controls
- Change the pair with the dropdown at the top. The **1Y–All** buttons change the period.
- Drag to zoom in and double-click to reset. All three charts share one time axis.

#### Caveats
- Returns ignore commissions, the bid-ask spread and borrowing costs for shorting, and trades are assumed to fill at the closing price.
- ±{ENTRY_Z:g}, ±{STOP_Z:g} and {WINDOW} days are fixed choices, not optimised. Panel 3 tests settings on unseen data.
"""

def build_panel2():
  gr.Markdown("""
              ## Panel 2 - Does the gap between rivals offer reliable trade signals?)
              """)
  with gr.Row():
    pair = gr.Radio(LABELS, value=LABELS[0], visible=False)   # hidden: the dashboard's top dropdown sets the pair
    with gr.Column(scale=5):
      plot = gr.Plot(make_figure(LABELS[0], "3Y"), show_label=False)
      period = gr.State("3Y")
      with gr.Row():
        buttons = [gr.Button(p, size="lg", min_width=0, variant="primary" if p == "3Y" else "secondary") for p in PERIODS]
    with gr.Column(scale=2, min_width=300):
      guide = gr.Markdown(explain(LABELS[0]), height=815, elem_id="guide")   # scrolls if the text is longer
  pair.change(make_figure, [pair, period], plot)
  pair.change(explain, pair, guide)
  for button in buttons:
    button.click(pick_period, [pair, button], [period, plot] + buttons)
  return pair                          # so the dashboard can change the pair

if __name__ == "__main__":
  print("Downloading prices...")
  print(f"Got {load_prices().shape[1]} tickers, {len(load_prices())} days. Building first chart... ")
  with gr.Blocks(title="Panel 2 - Spread & Trade Signals (Luke Kiely)") as demo: 
    build_panel2()
  demo.launch(theme=gr.themes.Default(primary_hue="blue"))
  

  

