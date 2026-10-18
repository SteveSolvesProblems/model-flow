"""Shared figure style: one script per figure, SVG output to publish/docs/figures/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE.parents[1] / "publish" / "docs" / "figures"
INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
BLUE, ORANGE, GREEN, PURPLE = "#0969da", "#d1700f", "#1a7f37", "#8250df"

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
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
    print("wrote", OUT / f"{name}.svg")
