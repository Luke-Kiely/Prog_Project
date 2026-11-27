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
CLOSE_DAYS = 20  # trading days a signal's gap is given to close (about a month)
CLOSE_Z = 0.5  # the gap counts as closed once it is back within 0.5 of its average
# look of each headline-number card above the charts
CARD = "flex:1; background:#1a1d23; border-radius:10px; padding:12px 16px; color:#e6e6e6"


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
    # z-score is how far today's spread (A minus the hedged amount of B) is from its
    # recent average, in standard deviations. Today's hedge ratio is used for the whole
    # window, so a change in the hedge ratio can't look like a price move.
    mean_a, mean_b = log_a.rolling(z_window).mean(), log_b.rolling(z_window).mean()
    gap = (log_a - mean_a) - hedge * (log_b - mean_b)  # today's spread minus its average
    spread_var = (log_a.rolling(z_window).var() + hedge ** 2 * log_b.rolling(z_window).var()
                  - 2 * hedge * log_a.rolling(z_window).cov(log_b))
    z_score = gap / np.sqrt(spread_var)

    # for each signal day, did the gap close within CLOSE_DAYS? That day's hedge ratio,
    # average and standard deviation are kept fixed, so only later prices can close it
    closed = pd.Series(False, index=z_score.index)
    for day in z_score.index[z_score.abs() >= SIGNAL_Z]:
        later_a = log_a.loc[day:].iloc[1:CLOSE_DAYS + 1]  # the next CLOSE_DAYS days
        later_b = log_b.loc[day:].iloc[1:CLOSE_DAYS + 1]
        later_gap = (later_a - mean_a[day]) - hedge[day] * (later_b - mean_b[day])
        later_z = later_gap / np.sqrt(spread_var[day]) * np.sign(z_score[day])
        closed[day] = (later_z <= CLOSE_Z).any()  # back near (or past) the average

    # daily returns and z-score (drops days without a z-score yet)
    df = pd.DataFrame({"ret_a": log_a.diff(), "ret_b": log_b.diff(), "z": z_score}).dropna()
    df["closed"] = closed  # True on signal days whose gap closed
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
    """Builds a table with one row per year: beta, R-squared and hover text
    for the bottom chart."""
    rows = []
    for year, group in df.groupby("year"):  # one group of days per year
        if len(group) < MIN_DAYS:
            continue  # skip years with too few days
        beta, _, r_squared = regression(group["ret_b"], group["ret_a"])  # _ skips alpha
        signal_days = (group["z"].abs() >= SIGNAL_Z).sum()  # days beyond signal threshold
        closed = group["closed"].sum()  # signal days whose gap closed
        # text shown when hovering over this year's bar
        hover = (f"{year}<br>Beta {beta:.2f}<br>R-squared {r_squared:.2f}<br>"
                 f"{describe(r_squared).title()}<br>Signal days {signal_days} of {len(group)}"
                 f"<br>Gaps that closed {closed} of {signal_days}")
        rows.append({"Year": year, "Beta": beta, "R-squared": r_squared, "Hover": hover})
    # keep the column names even if no years qualify
    return pd.DataFrame(rows, columns=["Year", "Beta", "R-squared", "Hover"])


def style_figure(fig: go.Figure, title: str, subtitle: str, height: int) -> go.Figure:
    """Applies the same styling to both charts."""
    # bold title with a smaller grey subtitle underneath
    title = (f"<b>{title}</b><br>"
             f"<span style='font-size:13px; color:#a0a4ab'>{subtitle}</span>")
    fig.update_layout(title=dict(text=title, font_size=20, x=0.02), template="plotly_dark",
                      font=dict(family="Arial", size=13, color="#e6e6e6"),  # all other text
                      hoverlabel=dict(font_size=13), legend=dict(font_size=13),
                      height=height, showlegend=False,
                      margin=dict(t=95, l=80, r=190))  # same side margins so both charts line up
    # turn off zoom, and make axis titles bigger than the tick numbers
    fig.update_xaxes(fixedrange=True, title_font_size=15, tickfont=dict(size=12, color="#a0a4ab"))
    fig.update_yaxes(fixedrange=True, title_font_size=15, tickfont=dict(size=12, color="#a0a4ab"))
    # faint gridlines so the data stands out
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.3)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.3)")
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
        title = f"{pair_name}, {year}: {describe(r_squared).title()}"
        subtitle = "Each dot is one trading day. Hover over a dot for its date and moves."

        # draw the most extreme days last so they sit on top
        this_year = this_year.sort_values("z", key=abs)
        # one dot per day, coloured by that day's spread z-score
        fig.add_trace(go.Scatter(
            x=this_year["ret_b"], y=this_year["ret_a"], mode="markers", name="Spread z-score",
            text=this_year.index.strftime("%d %b %Y"),  # the date, shown when hovering
            hovertemplate=(f"%{{text}}<br>{stock_a} %{{y:+.1%}}, {stock_b} %{{x:+.1%}}"
                           "<br>Spread z-score %{marker.color:+.2f}<extra></extra>"),
            marker=dict(size=9, opacity=0.85, color=this_year["z"], cmin=-Z_CAP, cmax=Z_CAP,
                        colorscale=[[0, "#2a78d6"], [0.5, "#8a919c"], [1, "#d03b3b"]],
                        # labels at the ends of the colour bar instead of a long title
                        colorbar=dict(title=dict(text="Spread z-score", side="right"), len=0.7,
                                      tickvals=[-Z_CAP, 0, Z_CAP],
                                      ticktext=[f"-3 ({stock_a} cheap)", "0 (normal)",
                                                f"+3 ({stock_a} expensive)"]),
                        line=dict(width=1, color="#111111")),  # thin dark outline
        ))

        # regression line across the whole chart
        line_x = np.array(axis_range)  # the two ends of the x-axis
        # faint dashed beta = 1 line: dots on it mean both stocks moved the same %
        fig.add_trace(go.Scatter(x=line_x, y=line_x, mode="lines", hoverinfo="skip",
                                 line=dict(color="#8a919c", width=1, dash="dash")))
        fig.add_annotation(x=0.9 * limit, y=0.9 * limit, text="beta = 1 ", showarrow=False,
                           font=dict(color="#8a919c", size=12), xanchor="right")
        fig.add_trace(go.Scatter(x=line_x, y=alpha + beta * line_x, mode="lines",
                                 name="Regression line", line=dict(color="white", width=2)))

    # axis labels as a percentage with the same range every year
    fig.update_xaxes(title=f"{stock_b} Daily Return", range=axis_range,
                     tickformat=".0%", hoverformat=".1%")
    fig.update_yaxes(title=f"{stock_a} Daily Return", range=axis_range,
                     tickformat=".0%", hoverformat=".1%")
    return style_figure(fig, title, subtitle, height=600)  # apply the shared chart styling


def make_history(df: pd.DataFrame, year: int) -> go.Figure:
    """Builds the bottom chart: beta (line) and R-squared (bars) for each year,
    with the year shown in the top chart picked out."""
    summary = yearly_summary(df)
    beta_all = regression(df["ret_b"], df["ret_a"])[0]  # beta over all years ([0] = beta)
    fig = go.Figure()

    chosen = summary[summary["Year"] == year]  # the year shown in the top chart

    # R-squared as bars, with the chosen year solid and the rest faded
    fig.add_trace(go.Bar(x=summary["Year"], y=summary["R-squared"], name="R-squared",
                         marker_color="#2a78d6", hovertext=summary["Hover"], hoverinfo="text",
                         marker_opacity=np.where(summary["Year"] == year, 1, 0.4)))
    # beta as a white line, the same colour as the regression line above
    fig.add_trace(go.Scatter(x=summary["Year"], y=summary["Beta"], mode="lines+markers",
                             name="Beta", line=dict(color="white"), hoverinfo="skip"))
    # big dot on the chosen year's beta: the slope of the line in the top chart
    fig.add_trace(go.Scatter(x=chosen["Year"], y=chosen["Beta"], mode="markers",
                             marker=dict(size=16, color="white"), showlegend=False,
                             hoverinfo="skip"))
    # dotted line showing beta across all years to compare each year against
    fig.add_hline(y=beta_all, line_dash="dot", line_color="#8a919c",
                  annotation_text=f"All-years beta {beta_all:.2f}",
                  annotation_position="top left", annotation_font_size=11)
    # same years on the x-axis for every pair, labelled every 2 years
    fig.update_xaxes(range=[FIRST_YEAR - 0.5, LAST_YEAR + 0.5], dtick=2)
    subtitle = "Big white dot = the slope of the line above. Hover over a bar for its numbers."
    fig = style_figure(fig, f"How {year} compares with every other year", subtitle, height=400)
    # show the legend (style_figure turns it off), with thinner, rounded bars
    fig.update_layout(showlegend=True, bargap=0.35, barcornerradius=4)
    return fig


def make_stats(df: pd.DataFrame, year: int) -> str:
    """Builds the row of headline numbers for the chosen year, shown above the charts."""
    this_year = df[df["year"] == year]
    if len(this_year) < MIN_DAYS:
        return "<p>No data for this year.</p>"
    beta, _, r_squared = regression(this_year["ret_b"], this_year["ret_a"])
    signal_days = (this_year["z"].abs() >= SIGNAL_Z).sum()  # days beyond signal threshold
    closed = this_year["closed"].sum()  # signal days whose gap closed
    stats = [("Beta", f"{beta:.2f}"), ("R-squared", f"{r_squared:.2f}"),
             ("Signal days", f"{signal_days} of {len(this_year)}"),
             (f"Gaps closed in {CLOSE_DAYS} days", f"{closed} of {signal_days}"),
             ("Relationship", describe(r_squared).title())]
    # one card per number: small grey name on top, big bold value underneath
    cards = "".join(f"<div style='{CARD}'><div style='color:#a0a4ab; font-size:13px'>{name}</div>"
                    f"<div style='font-size:26px; font-weight:bold'>{value}</div></div>"
                    for name, value in stats)
    # one-sentence verdict that answers "are they tethered?" for this pair
    summary = yearly_summary(df)
    rank = (summary["R-squared"] > r_squared).sum() + 1  # 1 = tightest year
    verdict = (f"Across {len(summary)} years, beta ranged {summary['Beta'].min():.2f} to "
               f"{summary['Beta'].max():.2f} and R-squared was above 0.5 in "
               f"{(summary['R-squared'] >= 0.5).sum()} of them, so overall the pair is "
               f"<b>{describe(summary['R-squared'].median())}</b>. {year} ranks "
               f"#{rank} of {len(summary)} for how tightly the two moved together. "
               f"Over all years, {df['closed'].sum() / (df['z'].abs() >= SIGNAL_Z).sum():.0%} "
               f"of signal gaps closed within {CLOSE_DAYS} trading days.")
    return (f"<div style='display:flex; gap:12px'>{cards}</div>"
            f"<p style='margin:14px 4px 0; font-size:15px; color:#e6e6e6'>{verdict}</p>")


def make_charts(pair_name: str, year: int, z_window: int) -> tuple:
    """Builds both charts and the headline numbers for the current pair, year and window."""
    # Gradio sliders give decimals so convert to whole numbers
    df = prepare_pair(pair_name, int(z_window))
    year = int(year)
    return make_scatter(df, pair_name, year), make_history(df, year), make_stats(df, year)


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
    first_scatter, first_history, first_stats = make_charts(list(PAIRS)[0], FIRST_YEAR, 60)
    # heading, the question, and a quick guide to the panel (always visible)
    gr.Markdown(f"""
## Panel 4: Is the link between two rivals stable from year to year?
How closely two rival stocks move together in each year from {FIRST_YEAR} to today.
Press **▶ Play** to step through the years, and hover over anything for exact values.
- **Cards:** this year's link in numbers: **beta** (how much A moves when B moves 1%) and
  **R-squared** (how linked the two are, from 0 to 1).
- **Top chart:** each dot is one trading day; the white line's slope is this year's beta.
  Red dots = A unusually expensive compared with B that day, blue = unusually cheap.
- **Bottom chart:** each blue bar is one year's R-squared; the white line is each year's
  beta. The big white dot marks the year shown in the top chart.

**How to read it:** if the bars and line stay at a similar level year after year, the pair
is reliably tethered. If they jump around, the link comes and goes, so a gap between the two
prices is a weaker trading signal.
""")
    # choose the pair and the z-score window
    with gr.Row():
        pair_dropdown = gr.Dropdown(list(PAIRS), value=list(PAIRS)[0], label="Rival pair")
        z_slider = gr.Slider(20, 120, value=60, step=10,
                             label="Spread z-score window (trading days)")
    # Play / Pause buttons and the year slider on one row
    with gr.Row(equal_height=True):
        play_button = gr.Button("▶ Play", variant="primary", scale=0, min_width=110)
        pause_button = gr.Button("⏸ Pause", scale=0, min_width=110)
        year_slider = gr.Slider(FIRST_YEAR, LAST_YEAR, value=FIRST_YEAR, step=1, label="Year")
    # headline numbers and both charts together in one box
    with gr.Column(variant="panel"):
        stats_box = gr.HTML(first_stats)
        # fixed heights stop the page jumping to the top while the charts redraw
        with gr.Row(min_height=620):
            scatter_plot = gr.Plot(first_scatter, show_label=False)
        with gr.Row(min_height=420):
            history_plot = gr.Plot(first_history, show_label=False)
    # the detail behind the guide above, folded away until clicked
    with gr.Accordion("More detail: the charts, data and method", open=False):
        gr.Markdown(f"""
**More on the charts**
- **Beta example:** 0.85 means stock A moved 0.85% on average for every 1% move in B.
- **R-squared:** 0 = no link, 1 = lockstep. The closer the top chart's dots hug the white
  line, the higher it is.
- **Dashed line:** beta = 1, where both stocks move by the same percentage.
- **Dot colour:** the spread z-score, meaning how far apart the two prices were that day
  compared with their recent normal.
- **Signal days:** days with a z-score beyond plus or minus {SIGNAL_Z}. **Gaps closed:** how
  many of those came back within {CLOSE_Z} of normal in the next {CLOSE_DAYS} trading days,
  keeping that day's hedge ratio fixed.

**Data**
- Daily closing prices adjusted for splits and dividends, from Yahoo Finance using the
  yfinance library, {START[:4]} to {LAST_YEAR} ({LAST_YEAR} is a partial year).
  Saved in `data/prices.csv` so the panel works without downloading.

**Method**
- Returns are daily changes in log prices; beta and R-squared come from a regression of
  stock A's returns on stock B's, one year at a time.
- Hedge ratio: slope of log A on log B over the previous {HEDGE_WINDOW} trading days,
  using yesterday's value so it never uses future data.
- Spread z-score: today's spread compared with its average over the chosen window,
  using today's hedge ratio for the whole window.
- Years with fewer than {MIN_DAYS} trading days are left out.

**Caveats**
- Gaps closed shows whether the gap tends to close, not whether a trade would have made
  money after costs (Panels 2 and 3 look at trading).
- Beta and R-squared measure how the stocks move day to day, not whether their prices
  stay together in the long run (Panel 1's cointegration test).
""")
    timer = gr.Timer(PLAY_SECONDS, active=False)  # ticks every second while playing, off at start

    # redraw charts when anything changes
    controls = [pair_dropdown, year_slider, z_slider]
    charts = [scatter_plot, history_plot, stats_box]
    pair_dropdown.change(make_charts, controls, charts)
    z_slider.release(make_charts, controls, charts)
    year_slider.change(make_charts, controls, charts)

    # Play/pause: the timer moves the year slider on, which redraws the charts
    play_button.click(start_playing, year_slider, [year_slider, timer])
    pause_button.click(stop_playing, None, timer)
    timer.tick(next_year, year_slider, [year_slider, timer])
    return pair_dropdown                # so the dashboard can change the pair


if __name__ == "__main__":
    with gr.Blocks(title="Panel 4 - Relationship over time") as demo:
        build_panel4()
    demo.launch()  # only runs when this file is executed directly
    