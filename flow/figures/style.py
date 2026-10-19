"""Shared figure style: one script per figure, SVG output to publish/docs/figures/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE.parents[1] / "publish" / "docs" / "figures"
# the pages' light palette; inline.py swaps these for the pages' CSS colour variables, so inlined figures follow the
# light and dark themes
INK, MUTED, GRID, BG = "#18211f", "#5b6763", "#d9dfdc", "#f3f5f4"
BLUE, ORANGE, GREEN, PURPLE = "#1f5fbf", "#c2582b", "#2f7d4f", "#8a5bb0"
CSS_VARS = {INK: "--ink", MUTED: "--muted", GRID: "--rule", BG: "--bg",
            BLUE: "--accent", ORANGE: "--warm", GREEN: "--good", PURPLE: "--c3"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "figure.dpi": 100,
})


def load(name):
    return json.loads((RESULTS / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160, facecolor=BG)
    print("wrote", OUT / f"{name}.svg")
