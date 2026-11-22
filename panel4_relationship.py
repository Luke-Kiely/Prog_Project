"""Panel 4 - Relationship over time (Charlie)."""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PRICE_FILE = Path("data") / "prices.csv"  # run from the project folder
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