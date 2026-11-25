# Group Name: LEMC Analysis
- Name: Luke Kiely
- Name: Eoin Hannon
- Name: Mathew Hirsch
- Name: Charles Graham

## How to run

**Requirements:** Python 3.10 or newer (tested on Python 3.13, macOS).

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

**Author:** [Your full name] · **File:** `panel2_spread_trade_signals.py`

### Purpose
Panel 1 asks whether rival stocks are tethered. Panel 2 asks what happens if you actually trade on that assumption. For each rival pair, it estimates a "fair value" for one stock based on its rival's price. It then flags when the gap between the two becomes unusually wide, and tracks the profit or loss of every trade a simple pairs strategy would have made.

The fair value comes from a rolling 60-day regression of log prices, using only data available up to the previous day, rather than simply rebasing both stocks to 100. This is deliberate: rivals rarely move one-for-one, and the regression estimates how they actually move together, so signals aren't triggered by normal differences between the two companies. The gap is then converted into a z-score so that "unusually wide" means the same thing across every pair and time period. Trades open at ±2 and close when the gap returns to zero, or at a ±3.5 stop-loss. Results are shown before trading costs, which Panel 3 then accounts for.