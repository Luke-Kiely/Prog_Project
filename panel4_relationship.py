"""Panel 4 - Relationship over time (Charlie)."""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PRICE_FILE = Path(__file__).parent / "data" / "prices.csv"  # works from any folder
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

# Check that all the tickers loaded successfully, if not then warn instead of crashing
missing = set(TICKERS) - set(PRICES.columns)  # tickers we need but don't have
if missing:
    print(f"Warning: {PRICE_FILE} is missing {sorted(missing)}. Those pairs won't load.")


def prepare_pair(pair_name: str, z_window: int) -> pd.DataFrame:
    """Daily log returns for both stocks, plus the spread z-score for each day."""
    stock_a, stock_b = PAIRS[pair_name]
    if stock_a in missing or stock_b in missing:
        # error only shows in this panel so the rest of the dashboard keeps working
        raise gr.Error(f"No price data for {stock_a} or {stock_b}.")
    both = PRICES[[stock_a, stock_b]].dropna()  # only days when both stocks traded
    # log prices so the changes are % returns
    log_a, log_b = np.log(both[stock_a]), np.log(both[stock_b])

    # Hedge ratio = slope of A on B over the last year
    # .shift(1) so the hedge ratio only uses past data
    hedge = (log_a.rolling(HEDGE_WINDOW).cov(log_b) / log_b.rolling(HEDGE_WINDOW).var()).shift(1)
    spread = log_a - hedge * log_b  # the gap between A and the hedged amount of B

    # z-score is how far today's spread is from its recent average, in standard deviations
    z_score = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()

    # daily returns and z-score (drops days without a z-score yet)
    df = pd.DataFrame({"ret_a": log_a.diff(), "ret_b": log_b.diff(), "z": z_score}).dropna()
    df["year"] = df.index.year  # used to split the data by year
    return df


def regression(x: pd.Series, y: pd.Series) -> tuple:
    """Return beta, alpha and R-squared for the line y = alpha + beta * x."""
    beta = y.cov(x) / x.var()  # slope: how much y moves when x moves
    alpha = y.mean() - beta * x.mean()  # intercept: y when x = 0
    r_squared = y.corr(x) ** 2  # share of y's movement explained by x
    return beta, alpha, r_squared


def describe(r_squared: float) -> str:
    """Turns an R-squared value into a simple label for the chart title."""
    if r_squared >= 0.5:
        return "tightly linked"
    elif r_squared >= 0.25:
        return "moderately linked"
    else:
        return "loosely linked"


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a table with one row per year: beta, R-squared and hover text for the bottom chart."""
    rows = []
    for year, group in df.groupby("year"):  # one group of days per year
        if len(group) < MIN_DAYS:
            continue  # skip years with too few days
        beta, _, r_squared = regression(group["ret_b"], group["ret_a"])  # _ skips alpha
        signal_days = (group["z"].abs() >= SIGNAL_Z).sum()  # days beyond signal threshold
        # text shown when hovering over this year's bar
        hover = (f"{year}<br>Beta {beta:.2f}<br>R-squared {r_squared:.2f}<br>"
                 f"{describe(r_squared).title()}<br>Signal days {signal_days} of {len(group)}")
        rows.append({"Year": year, "Beta": beta, "R-squared": r_squared, "Hover": hover})
    # keep the column names even if no years qualify
    return pd.DataFrame(rows, columns=["Year", "Beta", "R-squared", "Hover"])


def style_figure(fig: go.Figure, title: str, subtitle: str, height: int) -> go.Figure:
    """Applies the same styling to both charts."""
    fig.update_layout(title=f"{title}<br><sup>{subtitle}</sup>", template="plotly_white",
                      height=height, showlegend=False, margin=dict(t=90))  # top room for the title
    # turn off zoom on both axes
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def make_scatter(df: pd.DataFrame, pair_name: str, year: int) -> go.Figure:
    """Builds a scatter plot with one dot per trading day in the chosen year,
    plus that year's regression line."""
    stock_a, stock_b = PAIRS[pair_name]
    this_year = df[df["year"] == year]  # only the days in the chosen year
    fig = go.Figure()

    # axes are fixed so the chart doesn't jump
    limit = np.ceil(4 * df[["ret_a", "ret_b"]].std().max() * 100) / 100
    axis_range = [-limit, limit]

    if len(this_year) < MIN_DAYS:  # not enough data to plot this year
        title = f"{pair_name}, {year}"
        years = yearly_summary(df)["Year"]  # years that have enough data to plot
        subtitle = ("No data for this year. "
                    f"This pair has data from {years.min()} to {years.max()}.")
    else:  # enough data: work out the numbers for this year
        beta, alpha, r_squared = regression(this_year["ret_b"], this_year["ret_a"])
        signal_days = (this_year["z"].abs() >= SIGNAL_Z).sum()  # days beyond signal threshold
        title = f"{pair_name}, {year}: {describe(r_squared).title()}"
        subtitle = (f"Beta {beta:.2f}, R-squared {r_squared:.2f}, "
                    f"{signal_days} signal days out of {len(this_year)}")

        # one dot per day, coloured by that day's spread z-score
        fig.add_trace(go.Scatter(
            x=this_year["ret_b"], y=this_year["ret_a"], mode="markers", name="Spread z-score",
            text=this_year["z"].round(2),  # z-score shown when hovering
            marker=dict(size=8, color=this_year["z"], colorscale="RdBu_r", cmin=-Z_CAP, cmax=Z_CAP,
                        colorbar=dict(title=f"Spread z-score<br>Red = {stock_a} expensive<br>"
                                            f"Blue = {stock_a} cheap<br> "),
                        line=dict(width=0.5, color="grey")),  # outline so pale dots show
        ))

        # regression line across the whole chart
        line_x = np.array(axis_range)  # the two ends of the x-axis
        fig.add_trace(go.Scatter(x=line_x, y=alpha + beta * line_x, mode="lines",
                                 name="Regression line", line=dict(color="white", width=2)))

    # axis labels as a percentage with the same range every year
    fig.update_xaxes(title=f"{stock_b} Daily Return", range=axis_range,
                     tickformat=".0%", hoverformat=".1%")
    fig.update_yaxes(title=f"{stock_a} Daily Return", range=axis_range,
                     tickformat=".0%", hoverformat=".1%")
    return style_figure(fig, title, subtitle, height=600)  # apply the shared chart styling


def make_history(df: pd.DataFrame, year: int) -> go.Figure:
    """Builds the bottom chart: beta (line) and R-squared (bars) for each year."""
    summary = yearly_summary(df)
    beta_all = regression(df["ret_b"], df["ret_a"])[0]  # beta over all years ([0] = beta)
    fig = go.Figure()

    # R-squared as bars, with each year's numbers shown on hover
    fig.add_trace(go.Bar(x=summary["Year"], y=summary["R-squared"], name="R-squared",
                         hovertext=summary["Hover"], hoverinfo="text"))
    # beta as an orange line with a dot per year
    fig.add_trace(go.Scatter(x=summary["Year"], y=summary["Beta"], mode="lines+markers",
                             name="Beta", line=dict(color="orange")))
    # shade the year shown in the top chart
    fig.add_vrect(x0=year - 0.5, x1=year + 0.5, fillcolor="white", opacity=0.15, line_width=0)
    # dotted line showing beta across all years to compare each year against
    fig.add_hline(y=beta_all, line_dash="dot", annotation_text=f"All-years beta {beta_all:.2f}")
    # same years on the x-axis for every pair, labelled every 2 years
    fig.update_xaxes(range=[FIRST_YEAR - 0.5, LAST_YEAR + 0.5], dtick=2)
    subtitle = f"Shaded year: {year}. Hover over a bar for that year's numbers."
    fig = style_figure(fig, "Beta and R-squared by Year", subtitle, height=400)
    fig.update_layout(showlegend=True)  # show the legend here (it's off by default in style_figure)
    return fig


def make_charts(pair_name: str, year: int, z_window: int) -> tuple:
    """Builds both charts for the current pair, year and z-score window."""
    # Gradio sliders give decimals so convert to whole numbers
    df = prepare_pair(pair_name, int(z_window))
    return make_scatter(df, pair_name, int(year)), make_history(df, int(year))


# Play turns on a timer that moves the year slider
def start_playing(year: int) -> tuple:
    """Starts playing, going back to the start if already at the end."""
    if int(year) >= LAST_YEAR:
        year = FIRST_YEAR
    return year, gr.Timer(active=True)  # switch the timer on


def stop_playing():
    """Pauses by switching the timer off."""
    return gr.Timer(active=False)


def next_year(year: int) -> tuple:
    """Called by the timer: moves on one year, and stops at the last year."""
    year = int(year) + 1
    if year >= LAST_YEAR:
        return LAST_YEAR, gr.Timer(active=False)  # reached the end so stop playing
    return year, gr.Timer(active=True)


# Dashboard layout
def build_panel4():
    """Builds the panel inside whatever gr.Blocks calls it (the group dashboard or below)."""
    # draw the first pair's charts straight away so the panel isn't empty
    first_scatter, first_history = make_charts(list(PAIRS)[0], FIRST_YEAR, 60)
    # heading and short explanation at the top of the panel
    gr.Markdown("### Relationship over time\nPress **Play** to step through the years. "
                "**Beta** = how much stock A moves when stock B moves 1%. "
                "**R-squared** = how much of A's movement B explains (0 to 1). "
                "If both jump around from year to year, the stocks are not tethered.")
    # choose the pair and the z-score window
    with gr.Row():
        pair_dropdown = gr.Dropdown(list(PAIRS), value=list(PAIRS)[0], label="Rival pair")
        z_slider = gr.Slider(20, 120, value=60, step=10,
                             label="Spread z-score window (trading days)")
    # Play / Pause buttons
    with gr.Row():
        play_button = gr.Button("Play")
        pause_button = gr.Button("Pause")
    year_slider = gr.Slider(FIRST_YEAR, LAST_YEAR, value=FIRST_YEAR, step=1, label="Year")
    # the two charts, each in its own box
    with gr.Column(variant="panel"):
        scatter_plot = gr.Plot(first_scatter, show_label=False)
    with gr.Column(variant="panel"):
        history_plot = gr.Plot(first_history, show_label=False)
    timer = gr.Timer(PLAY_SECONDS, active=False)  # ticks every second while playing, off at start

    # redraw charts when anything changes
    controls = [pair_dropdown, year_slider, z_slider]
    charts = [scatter_plot, history_plot]
    pair_dropdown.change(make_charts, controls, charts)
    z_slider.release(make_charts, controls, charts)
    year_slider.change(make_charts, controls, charts)

    # Play/pause: the timer moves the year slider on, which redraws the charts
    play_button.click(start_playing, year_slider, [year_slider, timer])
    pause_button.click(stop_playing, None, timer)
    timer.tick(next_year, year_slider, [year_slider, timer])


if __name__ == "__main__":
    with gr.Blocks(title="Panel 4 - Relationship over time") as demo:
        build_panel4()
    demo.launch()  # only runs when this file is executed directly