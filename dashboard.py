""" Group Dashboard - Do Rival Stocks Make Good Pair Trades? """

import gradio as gr

from panel1_screener import build_panel1
from panel2_spread&trade_signals import build_panel2
from panel3_optimisation import build_panel3
from panel4_relationship import build_panel4

with gr.Blocks(title="Do rival stocks make good pair trades?") as demo: 
  gr.Markdown("# Do rival stocks make good pairs trades?\n"
              "Ten pairs of rival companies, tested as pairs trades.")
  with gr.Tab("1. Screener"):
    build_panel1()
  with gr.Tab("2. Spread & Trade Signals"):
    build_panel2()
  with gr.Tab("3. Optimisation"):
    build_panel3()
  with gr.Tab("4. Relationship"):
    build_panel4()

if __name__ == "__main__":
  demo.launch()



""" Each member must put this in
def build_panel3():
  gr.Plot(make_panel3_figure())

if __name__ == "__main__": 
  with gr.Blocks() as demo: 
    build_panel3()
  demo.launch()
"""
