""" Group Dashboard - Trading the Rival Gap """
import warnings

import gradio as gr
import plotly.io as pio

warnings.filterwarnings("ignore", module="yfinance")   # hide yfinance's pandas deprecation noise
pio.templates.default = "plotly_dark"                  # any chart without its own template matches the dark page

from panel1_screener import build_panel1
from panel2_spread_trade_signals import build_panel2
from panel3_optimisation import build_panel3
from panel4_relationship import build_panel4

TITLE = "Trading the Rival Gap"
SUBTITLE = "Can ten rival-stock pairs, from Coke vs Pepsi to Nvidia vs AMD, be traded for profit?"
OVERVIEW = """Are rival stocks genuinely tethered, and is the gap between them a reliable trading signal or a statistical illusion?

We test 10 pairs of rival companies, like Coca-Cola vs PepsiCo and Visa vs Mastercard, using daily prices from 2000 to today from Yahoo Finance.

**Work through the tabs in order:**
1. **Screener** - which of the 190 possible pairs of our 20 stocks are statistically tethered (cointegrated)?
2. **Spread & Trade Signals** - when the gap between two rivals stretches, where would a trader buy and sell, and did it pay off?
3. **Optimisation** - which strategy settings worked best in the past, and do they still work on later data they never saw?
4. **Relationship** - how closely do the two stocks move together, year by year?

Every chart is interactive: hover for exact numbers and use the controls to change the pair, dates and settings.
"""

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="zinc", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"]).set(
  body_background_fill_dark="#09090b", background_fill_primary_dark="#111111",
  background_fill_secondary_dark="#18181b", block_background_fill_dark="#111111",
  block_border_color_dark="#27272a", border_color_primary_dark="#27272a")
CSS = """
.gradio-container {width: 100% !important; max-width: 1320px !important; margin: 0 auto !important;}
#header {border: 1px solid #27272a; border-radius: 20px; padding: 44px 32px; text-align: center;
  background: radial-gradient(circle at 15% 0%, rgba(99,102,241,.35), transparent 55%),
              radial-gradient(circle at 90% 100%, rgba(34,211,238,.22), transparent 50%), #0f0f12;}
#header .kicker {color: #a5b4fc; letter-spacing: .2em; font-size: .75em; font-weight: 600; text-transform: uppercase;}
#header h1 {margin: 10px 0 0; font-size: 2.8em; font-weight: 800; letter-spacing: -.03em; color: transparent;
  background: linear-gradient(90deg, #fff, #c7d2fe 60%, #67e8f9); -webkit-background-clip: text; background-clip: text;}
#header p {color: #a1a1aa; margin: 12px 0 0; font-size: 1.1em;}
#nav {flex-wrap: nowrap; background: #111111; border: 1px solid #27272a; border-radius: 999px; padding: 6px; gap: 6px;}
#nav button {border-radius: 999px; font-weight: 600; min-width: 0; white-space: nowrap; border: none; box-shadow: none;}
#nav button.primary {background: linear-gradient(90deg, #6366f1, #22d3ee); color: #fff; box-shadow: 0 0 24px rgba(99,102,241,.45);}
#nav button.secondary {background: transparent; color: #a1a1aa;}
#source {text-align: center; color: #52525b; font-size: .85em;}
footer {display: none !important;}
"""


def build_overview():
  gr.Markdown(OVERVIEW)


PANELS = [("Introduction", build_overview), ("1. Screener", build_panel1), ("2. Spread & Trade Signals", build_panel2),
          ("3. Optimisation", build_panel3), ("4. Relationship", build_panel4)]


def show(chosen):
  """Show the chosen page, hide the others, and highlight its button."""
  sections = [gr.Column(visible=name == chosen) for name, _ in PANELS]
  buttons = [gr.Button(variant="primary" if name == chosen else "secondary") for name, _ in PANELS]
  return sections + buttons


with gr.Blocks(title=TITLE) as demo:
  gr.HTML(f"<div id='header'><div class='kicker'>Pairs trading study · ten rival pairs · 2000 to today</div>"
          f"<h1>{TITLE}</h1><p>{SUBTITLE}</p></div>")
  with gr.Row(elem_id="nav"):
    buttons = [gr.Button(name, variant="primary" if i == 0 else "secondary") for i, (name, _) in enumerate(PANELS)]
  sections = []
  for i, (name, build) in enumerate(PANELS):
    with gr.Column(visible=i == 0) as section:
      build()
    sections.append(section)
  gr.Markdown("Data: daily closing prices from Yahoo Finance via yfinance, 2000 to today.", elem_id="source")
  for button in buttons:
    button.click(show, button, sections + buttons)

if __name__ == "__main__":
  demo.launch(inbrowser=True, theme=THEME, css=CSS, js="() => document.body.classList.add('dark')")