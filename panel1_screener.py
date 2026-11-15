from pathlib import Path

import pandas as pd
import yfinance as yf

#The 10 rival pairs from our group plan
RIVAL_PAIRS=[
    ("KO", "PEP"),      # soft drinks
    ("V", "MA"),        # card networks
    ("XOM", "CVX"),     # oil majors
    ("JPM", "BAC"),     # big banks
    ("HD", "LOW"),      # home improvement
    ("WMT", "COST"),    # big-box retail
    ("MCD", "QSR"),     # fast food (QSR = Burger King's owner)
    ("UPS", "FDX"),     # parcel delivery
    ("NVDA", "AMD"),    # chips
    ("GOOGL", "META"),  # digital advertising
]

TICKERS= [ticker for pair in RIVAL_PAIRS for ticker in pair]

START= "2000-01-01"
END= None
PRICE_CACHE= Path("data") / "prices.csv"

print(TICKERS)