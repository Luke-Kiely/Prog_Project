# Group Name: LEMC Analysis
- Name: Luke
- Name: Eoin
- Name: Mathew
- Name: Charlie

## Panel 1: Rival Screener — *Which of the 190 pairs are genuinely tethered?*

**Author:** Eoin Hannon · **File:** `panel1_screener.py`

### Purpose
Pairs trading assumes that two related stocks are "tethered": when the gap between them widens, it tends to close again. Panel 1 tests that assumption before any trading is considered. It screens every possible pair of our 20 stocks (190 pairs) for a long-run relationship (cointegration) and shows the results as an interactive heatmap.

Testing all 190 pairs, rather than only our 10 rival pairs, is deliberate: the 180 unrelated pairs act as a control group. If rivals were genuinely tethered, they should pass the test more often than random pairs.
