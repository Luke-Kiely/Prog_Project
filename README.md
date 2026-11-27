# Group Name: LEMC Analysis
- Name: Luke Kiely
- Name: Eoin Hannon
- Name: Mathew Hirsch
- Name: Charles Graham

## How to run

**Requirements:** Python 3.11 or newer (tested on Python 3.13, macOS).

1. Clone the repository and open the folder:
   git clone https://github.com/Luke-Kiely/Prog_Project.git
   cd Prog_Project

2. Create and activate a virtual environment:
   - macOS / Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`

3. Install the libraries:
   pip install -r requirements.txt

4. Start the dashboard:
   python3 dashboard.py        (Windows: python dashboard.py)

The dashboard opens in your browser automatically. If it doesn't, open the address printed in the terminal (usually http://127.0.0.1:7860). The first load can take up to a minute while prices download from Yahoo Finance; if the download fails, the saved copy in `data/prices.csv` is used instead. Press Ctrl+C in the terminal to stop it.


## Project structure
| File | What it does | Author |
|---|---|---|
| dashboard.py | Combines the four panels into one app | ALL |
| panel1_screener.py | Cointegration screener for all 190 pairs | Eoin Hannon |
| panel2_spread_trade_signals.py | Fair value, z-score signals and trade P&L | Luke Kiely |
| panel3_optimisation.py | Strategy settings tested in and out of sample | Mathew Hirsch |
| panel4_relationship.py | Beta and R² of each pair over time | Charles Graham |
| data/prices.csv | Saved daily closing prices (fallback) | ALL |

## Data
Daily closing prices (adjusted) from Yahoo Finance via the yfinance library, 2000 to today.

## Panel 1: Rival Screener — *Which of the 190 pairs are genuinely tethered?*

**Author:** Eoin Hannon · **File:** `panel1_screener.py`

### Purpose
Pairs trading assumes that two related stocks are "tethered": when the gap between them widens, it tends to close again. Panel 1 tests that assumption before any trading is considered. It screens every possible pair of our 20 stocks (190 pairs) for a long-run relationship (cointegration) and shows the results as an interactive heatmap.

Testing all 190 pairs, rather than only our 10 rival pairs, is deliberate: the 180 unrelated pairs act as a control group. If rivals were genuinely tethered, they should pass the test more often than random pairs.


## Panel 2: Spread & Trade Signals — *When rivals diverge, where would a trader act, and did it pay?*

**Author:** Luke Kiely · **File:** `panel2_spread_trade_signals.py`

### Purpose
Panel 1 asks whether rival stocks are tethered. Panel 2 asks what happens if you actually trade on that assumption. For each rival pair, it estimates a "fair value" for one stock based on its rival's price. It then flags when the gap between the two becomes unusually wide, and tracks the profit or loss of every trade a simple pairs strategy would have made.

The fair value comes from a rolling 60-day regression of log prices, using only data available up to the previous day, rather than simply rebasing both stocks to 100. This is deliberate: rivals rarely move one-for-one, and the regression estimates how they actually move together, so signals aren't triggered by normal differences between the two companies. The gap is then converted into a z-score so that "unusually wide" means the same thing across every pair and time period. Trades open at ±2 and close when the gap returns to zero, or at a ±3.5 stop-loss. Results are shown before trading costs, which Panel 3 then accounts for.

## Panel 3: Strategy Optimisation - *Does the best setting still work on new data?*

**Author:** Mathew Hirsch · **File:** `panel3_optimisation.py`

### Purpose
Panel 2 trades one fixed rule. Panel 3 asks whether a better rule can be found, and whether it still works afterwards. For each rival pair it tests 36 settings of a simple pairs strategy: 6 entry thresholds (how far the gap must stretch before trading, from 0.5 to 3 standard deviations) and 6 lookback windows (how many past days count as "normal", from 10 to 250). It picks the setting with the highest Sharpe ratio on the first part of the data (in-sample, 70% of days by default), then runs that exact setting on the later days it has never seen (out-of-sample).

This is because if you test enough settings on the same data, one will almost always look good by luck, so the only fair test is on new data. To keep the test honest, the hedge ratio is fitted on in-sample days only, trades happen the day after a signal, every trade pays a cost, and a setting needs at least 10 in-sample trades to be picked. Two extra checks sit on top: a luck test, which flips the sign of each out-of-sample day's return 500 times (under 5% means probably not luck), and a split between calm and stressed markets using the VIX from the FRED API (a VIX of 20 or more counts as stressed).

### How to use it
- Pick a pair with the **Rival pair** selector at the top of the dashboard.
- Switch **Show** between **In-sample** and **Out-of-sample** to see the same 36 settings on each period.
- Move the **In-sample period** slider (50 to 90% of days) or the **Trading cost** slider (0 to 20 basis points per trade). The chart and table update automatically.
- Drag the 3D chart to rotate it and hover for each setting's exact Sharpe ratio. The white line marks the best in-sample setting, and the grey sheet is a Sharpe ratio of zero: blue above it made money, red below it lost money.
- The table shows the best setting, its Sharpe ratio in and out of sample, calm and stressed results, trade counts and the luck test.

### VIX data (FRED API key)
The VIX comes from the FRED API, which needs a free key from fred.stlouisfed.org. Set it before starting the dashboard:
- macOS / Linux: `export FRED_API_KEY=your_key`
- Windows: `set FRED_API_KEY=your_key`

Without a key, the panel uses the saved copy in `data/vix.csv`, so the dashboard still runs.

## Panel 4: Relationship over Time — *Is the link between two rivals stable from year to year?*

**Author:** Charles Graham · **File:** `panel4_relationship.py`

### Purpose
Panel 1 tests whether each pair is tethered over the whole period. Panel 4 asks whether that link actually holds year by year. A pair that moves closely together in some years but drifts apart in others is risky to trade, even if it passes a full-period test.

For each pair and each year from 2001, the panel regresses one stock's daily returns on its rival's. **Beta** shows how much stock A moves when stock B moves 1%, and **R-squared** shows how much of A's movement B explains (0 to 1). The top chart plots every trading day in the chosen year with that year's regression line; the bottom chart shows beta and R-squared for every year, so changes in the relationship stand out.

### How to use it
- Pick a pair with the **Rival pair** selector at the top of the dashboard.
- Press **▶ Play** to step through the years, or drag the **Year** slider. **⏸ Pause** stops it.
- Hover over any dot for its date and both stocks' moves, or over any bar for that year's numbers.
- **More detail** at the bottom of the panel explains the charts, data and method.


## Conclusion — *Is the gap between rivals a reliable signal, or a statistical illusion?*

**For a simple pairs strategy, mostly an illusion.** Some rivals clearly move together, but the gap between them is not a dependable trading signal.

- **Rivals are no more tethered than random pairs** *(Panel 1, all years, 5% level).* Only 3 of our 10 rival pairs pass the cointegration test, about the same rate as the 180 unrelated pairs (26%), and none survive the Bonferroni correction.
- **Signals can look profitable on paper** *(Panel 2, "All" period).* Before trading costs, adding up every trade's return gives a positive total for 6 of the 10 pairs, with 52–68% of trades winning, but large losses on pairs like Nvidia/AMD show what happens when the link breaks.
- **The best settings don't survive new data** *(Panel 3, default 70% in-sample and 5 basis point cost).* The best in-sample setting's Sharpe ratio fell out-of-sample for 9 of the 10 pairs, and none passed the luck test.
- **The link itself comes and goes** *(Panel 4).* How closely rivals move together changes a lot from year to year and is strongest in market crises, when most stocks move together anyway. Only 11–23% of unusually wide gaps closed within a month.

Rival stocks do share a relationship, but it is not stable enough, and the gaps do not close reliably enough, for a simple pairs strategy to make money after costs. We tested one simple strategy on ten pairs; more advanced approaches, such as combining several signals or adjusting for moves in the overall market, might do better.

*All figures use data up to November 2026.*