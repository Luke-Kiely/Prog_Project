""" Group Dashboard - Do Rival Stocks Make Good Pair Trades? """
import gradio as gr

from panel1_screener import build_panel1
from panel2_spread_trade_signals import build_panel2
from panel3_optimisation import build_panel3
from panel4_relationship import build_panel4

TITLE = "Trading the Rival Gap"

OVERVIEW = """
### Are rival stocks genuinely tethered, and is the gap between them a reliable trading signal or a statistical illusion?

We test 10 pairs of rival companies, like Coca-Cola vs PepsiCo and Visa vs Mastercard, using daily prices from 2000 to today from Yahoo Finance.

**Work through the tabs in order:**
1. **Screener** - which of the 190 possible pairs of our 20 stocks are statistically tethered (cointegrated)?
2. **Spread & Trade Signals** - when the gap between two rivals stretches, where would a trader buy and sell, and did it pay off?
3. **Optimisation** - which strategy settings worked best in the past, and do they still work on later data they never saw?
4. **Relationship** - how closely do the two stocks move together, year by year?

Every chart is interactive: hover for exact numbers and use the controls to change the pair, dates and settings.
"""

def build_overview():
  gr.Markdown(OVERVIEW)

PANELS = [("Introduction", build_overview), ("1. Screener", build_panel1), ("2. Spread & Trade Signals", build_panel2), ("3. Optimisation", build_panel3), ("4. Relationship", build_panel4)]

def show(chosen):
  sections = [gr.Column(visible=name == chosen) for name, _ in PANELS]
  buttons = [gr.Button(variant="primary" if name == chosen else "secondary") for name, _ in PANELS]
  return sections + buttons
  
with gr.Blocks(title=TITLE) as demo: 
  gr.HTML(f"<div style='background: #2a78d6; border-radius:10px; padding:28px; text-align:center'>"
          f"<h1 style='color:white; margin:0; font-size:2.2em'>{TITLE}</h1></div>")
  with gr.Row():
    buttons = [gr.Button(name, variant="primary" if i == 0 else "secondary") for i, (name, _) in enumerate(PANELS)]
  sections = []
  for i, (name, build) in enumerate(PANELS): 
    with gr.Column(visible=i == 0) as section:
      build()
    sections.append(section)
  for button in buttons:
    button.click(show, button, sections + buttons)

if __name__ == "__main__":
  demo.launch(inbrowser=True, theme=gr.themes.Default(primary_hue="blue"))
