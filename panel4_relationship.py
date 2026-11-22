"""Panel 4 - Relationship over time (Charlie)."""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

PRICE_FILE = Path("data") / "prices.csv"  # run from the project folder
START = "2000-01-01"  # download start date (one year before FIRST_YEAR for the hedge ratio)