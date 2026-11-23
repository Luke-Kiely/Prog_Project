"""Panel 3 - Strategy Optimisation (Mathew)

Tests 36 pairs trading settings on past prices, then checks whether the best one
still works on later prices. Run it with: python panel3_optimisation.py
"""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PRICE_FILE = Path(__file__).parent / "data" / "prices.csv"     # works from any folder

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
SHARPE_COLOURS = [[0, "#d03b3b"], [0.5, "#8a919c"], [1, "#2a78d6"]]   # red = losing, grey = flat, blue = winning


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
    print(f"{len(prices)} trading days, {prices.index[0]:%b %Y} to {prices.index[-1]:%b %Y}")
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
    move = spread - spread.shift(1)     # daily change in the spread
    costs = (held - held.shift(1)).abs() * cost_bps / 10000
    return held, move, costs


def get_sharpe(returns):
    """Return the annualised Sharpe ratio (risk-free rate taken as 0)."""
    if returns.std() == 0:
        return 0                    # no trades, avoid dividing by zero
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)


# Step 3: test every setting

def make_sharpe_table(spread, split, cost_bps):
    """Test every entry and window and return one row of results for each."""
    rows = []
    for window in WINDOWS:
        for entry in ENTRIES:
            held, move, costs = run_strategy(spread, window, entry, cost_bps)
            daily_return = held * move - costs
            new_trade = (held != 0) & (held.shift(1) == 0)
            rows.append({
                "window": window,
                "entry": entry,
                "in_sample": get_sharpe(daily_return.iloc[:split]),
                "out_of_sample": get_sharpe(daily_return.iloc[split:]),
                "trades_in": new_trade.iloc[:split].sum(),
                "trades_out": new_trade.iloc[split:].sum(),
            })
    return pd.DataFrame(rows)


def luck_test(spread, split, best, cost_bps, n_tries=500):
    """Return how often pure luck does as well as the best setting out-of-sample.

    Each try flips a coin for every day to decide if that day's result was a gain or a loss.
    Under 5% means the result is probably not luck.
    """
    held, move, costs = run_strategy(spread, int(best["window"]), best["entry"], cost_bps)
    daily_return = (held * move - costs).iloc[split:]
    real_sharpe = get_sharpe(daily_return)

    rng = np.random.default_rng(42)
    as_good = 0
    for _ in range(n_tries):
        coin = rng.choice([-1, 1], size=len(daily_return))
        if get_sharpe(daily_return * coin) >= real_sharpe:
            as_good += 1
    return as_good / n_tries


# Step 4: chart and summary table

def draw_surface(table, best, sample, title):
    """3D surface of the Sharpe ratio, with a line through the best in-sample setting."""
    column = "in_sample" if sample == "In-sample" else "out_of_sample"
    grid = table.pivot_table(index="window", columns="entry", values=column)
    limit = max(SHARPE_LIMIT, table["in_sample"].abs().max(), table["out_of_sample"].abs().max())

    fig = go.Figure()
    fig.add_trace(go.Surface(x=grid.columns, y=grid.index, z=grid.values, colorscale=SHARPE_COLOURS,
                             contours=dict(x=dict(show=True, color="#111111"), y=dict(show=True, color="#111111")),
                             cmin=-limit, cmax=limit, colorbar_title="Sharpe",
                             hovertemplate="Entry %{x} sd<br>Lookback %{y} days<br>Sharpe %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Surface(x=grid.columns, y=grid.index, z=np.zeros(grid.shape), opacity=0.25,
                             colorscale=[[0, "white"], [1, "white"]], showscale=False, hoverinfo="skip"))   # Sharpe = 0 plane
    fig.add_trace(go.Scatter3d(x=[best["entry"]] * 2, y=[best["window"]] * 2, z=[-limit, limit],
                               mode="lines", line=dict(color="white", width=8),
                               name="Best in-sample setting"))   # vertical line, never hidden
    fig.update_layout(title=title, height=650, template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0),
                      scene=dict(xaxis_title="Entry (z-score)", yaxis_title="Lookback (days)",
                                 zaxis_title="Sharpe", zaxis_range=[-limit, limit],
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))))
    return fig


def make_summary(best, hedge_ratio, luck):
    """Return a small table with the results for the best in-sample setting."""
    return pd.DataFrame({
        "Measure": ["Best entry threshold", "Best lookback window", "Hedge ratio",
                    "Sharpe in-sample", "Sharpe out-of-sample",
                    "Trades in-sample", "Trades out-of-sample",
                    "Chance luck does as well (out-of-sample)"],
        "Value": [f"{best['entry']} sd", f"{best['window']:.0f} days", f"{hedge_ratio:.2f}",
                  f"{best['in_sample']:.2f}", f"{best['out_of_sample']:.2f}",
                  f"{best['trades_in']:.0f}", f"{best['trades_out']:.0f}",
                  f"{luck:.0%}"],
    })


def update_panel(pair_name, sample, in_sample_pct, cost_bps):
    """Runs every time a control changes and returns the chart and summary table."""
    prices = all_prices[PAIRS[pair_name]].dropna()      # days where both stocks traded
    split = int(len(prices) * in_sample_pct / 100)
    spread, hedge_ratio = get_spread(prices, split)
    table = make_sharpe_table(spread, split, cost_bps)
    enough_trades = table[table["trades_in"] >= MIN_TRADES]    # ignore settings that barely traded
    if enough_trades.empty:                                      # no setting traded enough
        enough_trades = table
    best = enough_trades.loc[enough_trades["in_sample"].idxmax()]
    luck = luck_test(spread, split, best, cost_bps)

    change = "falls" if best["out_of_sample"] < best["in_sample"] else "rises"
    title = (f"{pair_name}: best in-sample Sharpe of {best['in_sample']:.2f} "
             f"{change} to {best['out_of_sample']:.2f} after {prices.index[split]:%b %Y}")
    fig = draw_surface(table, best, sample, title)
    return fig, make_summary(best, hedge_ratio, luck)


# Main body

all_prices = load_prices()

first_fig, first_summary = update_panel("KO vs PEP", "In-sample", 70, 5)   # shown when the page opens

panel3_app = gr.Interface(
    fn=update_panel,
    inputs=[gr.Dropdown(list(PAIRS), value="KO vs PEP", label="Rival pair"),
            gr.Radio(["In-sample", "Out-of-sample"], value="In-sample", label="Show"),
            gr.Slider(50, 90, value=70, step=5, label="In-sample period (% of days)"),
            gr.Slider(0, 20, value=5, step=1, label="Trading cost per trade (basis points)")],
    outputs=[gr.Plot(value=first_fig, label="Sharpe ratio for every setting"),
             gr.Dataframe(value=first_summary, label="Best in-sample setting")],
    title="Panel 3: Does the best strategy still work out-of-sample?",
    live=True,
    flagging_mode="never",
    description="Drag to rotate the chart and hover for exact values. White line = best "
                "in-sample setting. Source: Yahoo Finance via yfinance.",
)


def build_panel3():
    """Panel 3 for the dashboard: draws the Interface above inside the dashboard page."""
    panel3_app.render()


if __name__ == "__main__":
    print("=== PANEL 3 CHECK: KO vs PEP, 70% in-sample, 5 bps cost ===")
    print(first_summary.to_string(index=False))
    panel3_app.launch()
