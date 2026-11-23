""" Group Dashboard - Do Rival Stocks Make Good Pair Trades? """
import importlib
import gradio as gr

from panel1_screener import build_panel1
from panel2_spread_trade_signals import build_panel2
from panel3_optimisation import build_panel3
from panel4_relationship import build_panel4

PANELS = [("1. Screener", build_panel1), ("2. Spread & Trade Signals", build_panel2), ("3. Optimisation", build_panel3), ("4. Relationship", build_panel4)]

def show(chosen):
  sections = [gr.Column(visible=name == chosen) for name, _ in PANELS]
  buttons = [gr.Button(variant="primary" if name == chosen else "secondary") for name, _ in PANELS]
  return sections + buttons
  
with gr.Blocks(title="Do Rival Stocks Make Good Pair Trades?") as demo: 
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


""" Each member must put this in
def build_panel3():
  gr.Plot(make_panel3_figure())

if __name__ == "__main__": 
  with gr.Blocks() as demo: 
    build_panel3()
  demo.launch()
"""